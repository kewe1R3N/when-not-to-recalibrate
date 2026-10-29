#!/usr/bin/env python3
"""
AIC2 — Corridor-anchored conformal recalibration experiments (Paper 2).

Reuses the frozen VQTransAE win65 model + scoring code from
aic_hi_validation_win65.py. Implements:
  M0  reproduction gate  : replicate Paper-1 N St frozen detection tau (~0.659)
  M1  drift quantification: frozen thresholds calibrated on a SOURCE corridor,
                            applied unchanged to shifted contexts (season/device),
                            measure corridor false-alarm rate (FPR) vs 1% target.

Protocol (matches Paper-1 rebuttal `nst_holdout_calibration.py`):
  - raw per-window E_t (recon error) and D_t (codebook distance) from the model
  - FROZEN reference normalization: E0,E1,D0,D1 = min,max of E,D on SOURCE
    corridor-normal windows  ->  s_t = 0.5*(E-E0)/(E1-E0) + 0.5*(D-D0)/(D1-D0)
  - CDI = rolling-median smooth (k=5, window 11) of s_t
  - detection tau = P99 of smoothed s_t on SOURCE corridor-normal windows
  - FPR on a context = fraction of that context's corridor-normal windows whose
    smoothed s_t exceeds the frozen tau  (no labels needed on the corridor)

All thresholds FROZEN from the source; nothing is fit on the evaluation contexts.
Compute is trivial (1.8M params). Env: ptv3.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

# --- import the Paper-1 model + scoring code (import-safe: main() is guarded) ---
HERE = Path(__file__).resolve().parent
CODE = HERE.parent                      # .../master_project/code
sys.path.insert(0, str(CODE))
import torch
import aic_hi_validation_win65 as A     # noqa: E402

CONFIG = A.config
WIN, STEP, FS = CONFIG.WIN_SIZE, CONFIG.STEP, CONFIG.FS
K_SMOOTH = CONFIG.K_SMOOTH
ALPHA = 0.01                            # target false-alarm budget (1%)

CKPT = str(CODE / "code_version_sep18" / "window_size_ablation" / "win_65" / "model.pth")
LORO_NST = str(CODE / "latex_output" / "loro_results" / "fold_4_N" / "data" / "test.csv")
IPHONE = CODE / "anomaly_lable" / "iphonedata"
ANDROID_NST = str(CODE.parent / "infer" / "device_comparison" / "android_Nst.csv")
OUT = Path("/home/kewei/paper_polish/refine-logs/aic2_results")
OUT.mkdir(parents=True, exist_ok=True)


def load_xyz(path):
    """Load a ride CSV -> (data[N,3] float32, anomaly[N] or None)."""
    df = pd.read_csv(path, low_memory=False)
    for c in ("X", "Y", "Z"):
        if c not in df.columns:
            raise ValueError(f"{path}: missing column {c}")
    df[["X", "Y", "Z"]] = df[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    data = df[["X", "Y", "Z"]].values.astype(np.float32)
    anom = None
    if "anomaly" in df.columns:
        anom = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).values.astype(int)
    return data, anom


def window_normal_mask(anom, n_windows):
    """A window is normal iff ALL its points have anomaly==0 (any-anomaly => anomalous)."""
    if anom is None:
        return np.ones(n_windows, dtype=bool)          # unlabeled corridor => all normal
    m = np.zeros(n_windows, dtype=bool)
    for i in range(n_windows):
        s = i * STEP
        m[i] = not np.any(anom[s:s + WIN] == 1)
    return m


def zscore(data):
    """Per-ride per-axis standardization at MODEL ENTRY — matches Paper-1's own
    inference convention (inference_new_data.py L229-233: mean/std per axis over
    the ride, then (x-mean)/std). The win65 model was trained on standardized
    inputs (vqtransae_results/processed_data, exactly mean=0/std=1 per axis);
    feeding raw g-units is an input-convention fault (discovered 2026-07-07:
    it alone produced the v1 '0-52x drift' and the spring rho collapse).
    Physical proxies (VDV, crest/SHI) must keep using RAW data — only the
    model input is standardized."""
    return ((data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)).astype(np.float32)


def raw_ED(model, data, device, standardize=True):
    """Raw (un-normalized) per-window E_t, D_t via the Paper-1 forward pass.
    Input is z-scored here (single choke point for all aic2 scripts).
    standardize=False simulates the 'missing-standardization' pipeline fault
    (fault-injection experiments only)."""
    feed = zscore(data) if standardize else np.asarray(data, np.float32)
    sc = A.compute_scores_from_model(model, feed, device)
    return np.asarray(sc["E_t"], float), np.asarray(sc["D_t"], float)


def crest_per_window(data, n_windows):
    """SHI crest factor max|a|/RMS(a) on the 3-axis magnitude, per window."""
    mag = np.sqrt((data ** 2).sum(axis=1))
    cf = np.full(n_windows, np.nan)
    for i in range(n_windows):
        w = mag[i * STEP: i * STEP + WIN]
        rms = np.sqrt(np.mean(w ** 2))
        if rms > 1e-9:
            cf[i] = np.max(np.abs(w)) / rms
    return cf


def smooth(s):
    win = 2 * K_SMOOTH + 1
    return pd.Series(s).rolling(window=win, center=True, min_periods=1).median().values


def conformal_tau(vals, alpha=ALPHA):
    """Split-conformal order statistic, EXACTLY Eq.~(conformal) in main.tex:
    k = ceil((n+1)(1-alpha)); tau = v_(k) (1-based, ascending); +inf if k>n.
    This is NOT np.percentile (which linearly interpolates order statistics
    and does not carry the finite-sample (n+1) correction the paper cites)."""
    v = np.sort(np.asarray(vals, float))
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(v[k - 1])


# Guard band needed so that the closest cal/eval RETURNED SCORES share zero
# raw accelerometer samples -- not just zero window indices, and not just
# "no smoothing-index overlap" (an earlier, insufficient version of this
# derivation stopped there). Each returned score is a K_SMOOTH-radius
# rolling-median of RAW per-window E/D values, so the last cal score (at
# window index h-g-1) actually pools raw windows up to index h-g-1+K_SMOOTH,
# and the first eval score (at h+g) pools raw windows down to h+g-K_SMOOTH.
# The closest constituent RAW WINDOWS are therefore only
#   (h+g-K_SMOOTH) - (h-g-1+K_SMOOTH) = 2g+1-2*K_SMOOTH
# index-units apart, not 2g+1. Window i spans raw samples [i*STEP,i*STEP+WIN),
# so two windows i<j share no raw sample iff j-i >= ceil(WIN/STEP) = R. Raw
# independence of the two returned scores therefore needs
#   2g+1-2*K_SMOOTH >= R  =>  g >= (R+2*K_SMOOTH-1)/2.
_R = int(np.ceil(WIN / STEP))                                  # = 13 for WIN=65,STEP=5
_MIN_RAW_GUARD = -(-(_R + 2 * K_SMOOTH - 1) // 2)              # ceil(...); = 11 for WIN=65,STEP=5,K_SMOOTH=5
RAW_INDEP_GUARD = _MIN_RAW_GUARD + 1                            # +1 window of margin = 12 for WIN=65,STEP=5


def halves(n, guard=None):
    """Chronological calibration/evaluation split of a same-ride window index
    range, with a guard band of `guard` windows dropped on each side of the
    boundary (default RAW_INDEP_GUARD) so that (a) no returned window's
    centered rolling-median smoothing support (radius K_SMOOTH) crosses the
    boundary, AND (b) the closest cal/eval window pair shares zero raw
    accelerometer samples, not just zero window indices. Returns two disjoint
    boolean masks (neither covers the dropped guard band)."""
    if guard is None:
        guard = RAW_INDEP_GUARD
    h = n // 2
    cal = np.zeros(n, bool)
    cal[: max(0, h - guard)] = True
    tst = np.zeros(n, bool)
    tst[min(n, h + guard):] = True
    return cal, tst


def calibrate(E, D, normal_mask, cf=None):
    """Freeze normalization + detection tau from source corridor-normal windows."""
    En, Dn = E[normal_mask], D[normal_mask]
    E0, E1 = np.nanmin(En), np.nanmax(En)
    D0, D1 = np.nanmin(Dn), np.nanmax(Dn)
    fr = dict(E0=float(E0), E1=float(E1), D0=float(D0), D1=float(D1))
    st = 0.5 * (E - E0) / (E1 - E0 + 1e-8) + 0.5 * (D - D0) / (D1 - D0 + 1e-8)
    sm = smooth(st)
    fr["tau"] = conformal_tau(sm[normal_mask])                       # Eq.~(conformal)
    fr["tau_cdi"] = float(np.nanpercentile(sm[normal_mask], 90))     # top-decile CDI
    if cf is not None:
        fr["tau_shi"] = float(np.nanpercentile(cf[normal_mask], 90))  # top-decile SHI
    return fr


def cal_raw_slice(data, n_cal_windows):
    """Raw-sample prefix spanned by the first n_cal_windows windows only —
    use this (never the full ride) when fitting anything (e.g. a refit
    codebook) that must not see test-half samples."""
    if n_cal_windows <= 0:
        return data[:0]
    end = (n_cal_windows - 1) * STEP + WIN
    return data[:end]


def apply_frozen(E, D, fr):
    st = 0.5 * (E - fr["E0"]) / (fr["E1"] - fr["E0"] + 1e-8) + \
         0.5 * (D - fr["D0"]) / (fr["D1"] - fr["D0"] + 1e-8)
    return smooth(st)


def fpr_ci(exceed_bool):
    """Empirical FPR + Wilson 95% CI (binomial)."""
    x = np.asarray(exceed_bool, bool)
    n = x.size
    k = int(x.sum())
    p = k / n if n else float("nan")
    if n == 0:
        return p, (float("nan"), float("nan")), n
    z = 1.96
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, (max(0, centre - half), min(1, centre + half)), n


def score_ride(model, path, device, tag):
    data, anom = load_xyz(path)
    n = (len(data) - WIN) // STEP + 1
    E, D = raw_ED(model, data, device)
    nm = window_normal_mask(anom, n)
    cf = crest_per_window(data, n)
    return dict(tag=tag, path=str(path), data=data, anom=anom, n=n,
                E=E, D=D, normal_mask=nm, cf=cf)


# ---------------------------------------------------------------- M0
def run_M0(model, device):
    print("\n[M0] Reproduction gate — Paper-1 N St frozen tau")
    r = score_ride(model, LORO_NST, device, "loro_Nst")
    nnorm = int(r["normal_mask"].sum())
    fr = calibrate(r["E"], r["D"], r["normal_mask"], r["cf"])
    sm = apply_frozen(r["E"], r["D"], fr)
    fpr, ci, nn = fpr_ci(sm[r["normal_mask"]] > fr["tau"])
    res = dict(stage="M0", n_windows=r["n"], n_normal=nnorm,
               tau=fr["tau"], tau_cdi=fr["tau_cdi"], tau_shi=fr["tau_shi"],
               E0=fr["E0"], E1=fr["E1"], D0=fr["D0"], D1=fr["D1"],
               in_sample_FPR_on_normal=fpr,
               target_ref={"paper1_tau": 0.659, "paper1_tau_cdi": 0.43,
                           "paper1_tau_shi": 4.79, "paper1_n_normal": 1026})
    print(f"   windows={r['n']}  normal={nnorm}")
    print(f"   detection tau (P99, frozen)        = {fr['tau']:.4f}   (Paper-1: 0.659)")
    print(f"   tau_CDI (P90)                      = {fr['tau_cdi']:.4f} (Paper-1: 0.43)")
    print(f"   tau_SHI crest (P90)                = {fr['tau_shi']:.4f} (Paper-1: 4.79)")
    print(f"   in-sample FPR on normal @tau       = {fpr:.4f} (expect ~0.01)")
    (OUT / "M0_reproduction.json").write_text(json.dumps(res, indent=2))
    print(f"   -> {OUT/'M0_reproduction.json'}")
    return res


# ---------------------------------------------------------------- M1
def run_M1(model, device, source_campaign="summer"):
    print(f"\n[M1] Drift quantification — SOURCE corridor = {source_campaign}/Nst")
    # source corridor ride (unlabeled -> all windows corridor-normal); split-half
    src = score_ride(model, str(IPHONE / source_campaign / "Nst.csv"), device,
                     f"{source_campaign}_Nst_src")
    cal, tst = halves(src["n"])                # guard-banded: no smoothing-window bleed
    fr = calibrate(src["E"], src["D"], cal, src["cf"])
    sm_src = apply_frozen(src["E"], src["D"], fr)
    p_in, ci_in, n_in = fpr_ci(sm_src[tst] > fr["tau"])

    rows = [dict(axis="source", context=f"{source_campaign}/Nst(held-out half)",
                 n_windows=int(tst.sum()), FPR=p_in, ci_lo=ci_in[0], ci_hi=ci_in[1],
                 inflation=p_in / ALPHA)]

    # season axis: other campaigns' N St corridor
    for camp in ["spring", "summer", "autumn", "winter"]:
        if camp == source_campaign:
            continue
        f = IPHONE / camp / "Nst.csv"
        if not f.exists():
            continue
        r = score_ride(model, str(f), device, f"{camp}_Nst")
        sm = apply_frozen(r["E"], r["D"], fr)
        p, ci, nn = fpr_ci(sm[r["normal_mask"]] > fr["tau"])
        rows.append(dict(axis="season", context=f"{camp}/Nst", n_windows=nn,
                         FPR=p, ci_lo=ci[0], ci_hi=ci[1], inflation=p / ALPHA))

    # device axis: Android same corridor (Lincoln N St)
    if Path(ANDROID_NST).exists():
        r = score_ride(model, ANDROID_NST, device, "android_Nst")
        sm = apply_frozen(r["E"], r["D"], fr)
        p, ci, nn = fpr_ci(sm[r["normal_mask"]] > fr["tau"])
        rows.append(dict(axis="device", context="android/Nst", n_windows=nn,
                         FPR=p, ci_lo=ci[0], ci_hi=ci[1], inflation=p / ALPHA))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "M1_drift_fpr.csv", index=False)
    meta = dict(stage="M1", source_campaign=source_campaign, target_alpha=ALPHA,
                frozen=fr, rows=rows)
    (OUT / "M1_drift.json").write_text(json.dumps(meta, indent=2))
    print("\n   Corridor FPR under frozen thresholds (target = 0.01):")
    print(df.to_string(index=False,
          columns=["axis", "context", "n_windows", "FPR", "inflation"]))
    print(f"\n   -> {OUT/'M1_drift_fpr.csv'}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["M0", "M1", "both"], default="both")
    ap.add_argument("--source", default="summer")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={CKPT}")
    model = A.load_model(CKPT, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   params={n_params:,}")
    if args.stage in ("M0", "both"):
        run_M0(model, device)
    if args.stage in ("M1", "both"):
        run_M1(model, device, source_campaign=args.source)


if __name__ == "__main__":
    main()
