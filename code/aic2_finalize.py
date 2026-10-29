#!/usr/bin/env python3
"""
Block B finalize: (1) block-bootstrap CIs on the drift + recalibration FPRs,
(2) Charleston (Ravenel Bridge) compound cross-city+device+app shift context.

Charleston honesty note: Ravenel data is a DIFFERENT logger app — m/s^2 (not g),
gravity on the Y axis (not Z), ~105 Hz (not 10). We convert to g (/9.81), keep a
consistent orientation, and resample to 10 Hz. It is ONE bridge ride, so we split
it into contiguous segments and use the smoothest as the local corridor. This
bundles city+device+app/orientation shift => a "maximally different deployment"
stress test for the recalibration method, NOT a clean city-only axis. Citadel
rides are excluded (gravity-removed preprocessing = yet another convention).

SUPERSEDED (2026-09-08): part (2) below does NOT do the per-sample gravity-stream
reorientation the manuscript describes for Ravenel ("reoriented per sample with
its independent gravity stream", main.tex Sec. Implementation) — load_ravenel_g()
here only rescales by /9.81 with no rotation. aic2_city.py's Ravenel block (search
"gravity reorientation") is the correct, currently-cited source (its ranking
rho=+0.852 matches the manuscript's "0.85"); this file's Ravenel FPR/rho numbers
below are computed on mis-oriented input and must NOT be cited anywhere. Part (1)
(Lincoln season/device block-bootstrap CIs) is unaffected and remains the live
source for Figure~\ref{fig:recal}(b).
"""
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from scipy.signal import resample_poly
import aic2_exp as X
import aic2_codebook as CB

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.01
RAV = str(X.CODE / "code_version_sep18" / "New_Android_Ravenel_Bridge" / "TotalAcceleration.csv")
RNG = np.random.RandomState(42)


def block_bootstrap_ci(exceed, L=20, B=2000):
    """Percentile CI for mean of a boolean series, moving-block bootstrap."""
    e = np.asarray(exceed, float); n = e.size
    if n < L:
        p = e.mean(); return p, p, p
    nb = int(np.ceil(n / L))
    starts_max = n - L
    means = np.empty(B)
    for b in range(B):
        idx = RNG.randint(0, starts_max + 1, size=nb)
        samp = np.concatenate([e[s:s + L] for s in idx])[:n]
        means[b] = samp.mean()
    return float(e.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score_ride(model, data):
    n = (len(data) - X.WIN) // X.STEP + 1
    E, D = X.raw_ED(model, data, dev)
    V = np.full(n, np.nan)
    for i in range(n):
        s = i * X.STEP; w = data[s:s + X.WIN]
        V[i] = X.A.compute_window_proxies(w[:, 0], w[:, 1], w[:, 2], X.FS,
                                          X.A.config.HP_CUTOFF)["VDV_w"]
    return dict(n=n, E=E, D=D, V=V, data=data)


def load_ravenel_g():
    r = pd.read_csv(RAV)
    dur = r["seconds_elapsed"].values[-1] - r["seconds_elapsed"].values[0]
    rate = len(r) / dur
    xyz = r[["x", "y", "z"]].values.astype(np.float64) / 9.81      # m/s^2 -> g
    xyz10 = resample_poly(xyz, up=10, down=int(round(rate)), axis=0)
    return xyz10.astype(np.float32)


def main():
    m = X.A.load_model(X.CKPT, dev)
    orig_cb = m.quant.embed.weight.detach().clone()

    # ---- source (summer Lincoln) frozen calibration ----
    src = score_ride(m, X.load_xyz(str(X.IPHONE / "summer" / "Nst.csv"))[0])
    scal, _ = X.halves(src["n"])
    fr_src = X.calibrate(src["E"], src["D"], scal)

    # ===== (1) block-bootstrap CIs: frozen drift vs conformal recal =====
    print("=== (1) Block-bootstrap 95% CIs on corridor FPR (target=0.01) ===")
    ci_rows = []
    for axis, ctx, path in [("season", "spring", str(X.IPHONE / "spring" / "Nst.csv")),
                            ("season", "autumn", str(X.IPHONE / "autumn" / "Nst.csv")),
                            ("season", "winter", str(X.IPHONE / "winter" / "Nst.csv")),
                            ("device", "android", X.ANDROID_NST)]:
        r = score_ride(m, X.load_xyz(path)[0])
        cal, tst = X.halves(r["n"])          # guard-banded: no smoothing-window bleed
        sm = X.apply_frozen(r["E"], r["D"], fr_src)
        ex_f = (sm[tst] > fr_src["tau"]).astype(float)
        pf, lof, hif = block_bootstrap_ci(ex_f)
        tau_c = X.conformal_tau(sm[cal], ALPHA)
        ex_c = (sm[tst] > tau_c).astype(float)
        pc, loc, hic = block_bootstrap_ci(ex_c)
        ci_rows.append(dict(axis=axis, context=ctx,
                            frozen_FPR=round(pf, 4), frozen_CI=f"[{lof:.4f},{hif:.4f}]",
                            conformal_FPR=round(pc, 4), conformal_CI=f"[{loc:.4f},{hic:.4f}]"))
        print(f"  {ctx:8s} frozen {pf:.4f} CI[{lof:.4f},{hif:.4f}]   "
              f"conformal {pc:.4f} CI[{loc:.4f},{hic:.4f}]")
    pd.DataFrame(ci_rows).to_csv(X.OUT / "B_bootstrap_ci.csv", index=False)

    # ===== (2) Charleston (Ravenel) compound-shift context =====
    print("\n=== (2) Charleston / Ravenel (compound city+device+app shift) ===")
    rav = load_ravenel_g()
    R = score_ride(m, rav)
    print(f"  Ravenel resampled windows={R['n']}")
    # split into contiguous segments; smoothest (lowest median VDV) = local corridor
    nseg = 6
    seg = np.array_split(np.arange(R["n"]), nseg)
    medV = [np.nanmedian(R["V"][s]) for s in seg]
    corr_i = int(np.argmin(medV))
    corr = seg[corr_i]

    # Guard-band trim: corr is fit on (calibration/codebook) and `others` is the
    # independent evaluation set, so any window whose centered rolling-median
    # smoothing support (radius X.K_SMOOTH) crosses the corr<->neighbor-segment
    # boundary must be dropped from BOTH sides, not assigned to either.
    guard = X.K_SMOOTH
    corr_trim = corr
    if corr_i > 0 and len(corr_trim) > guard:
        corr_trim = corr_trim[guard:]
    if corr_i < nseg - 1 and len(corr_trim) > guard:
        corr_trim = corr_trim[:-guard]
    others_parts = []
    for i in range(nseg):
        if i == corr_i:
            continue
        s = seg[i]
        if i == corr_i - 1 and len(s) > guard:
            s = s[:-guard]
        if i == corr_i + 1 and len(s) > guard:
            s = s[guard:]
        others_parts.append(s)
    others = np.concatenate(others_parts)
    ch, chalf = corr_trim[:len(corr_trim) // 2], corr_trim[len(corr_trim) // 2:]   # corridor cal / test

    sm = X.apply_frozen(R["E"], R["D"], fr_src)
    # frozen (Lincoln thresholds) on Charleston non-corridor segments
    pf, lof, hif = block_bootstrap_ci((sm[others] > fr_src["tau"]).astype(float))
    # conformal recal on Charleston corridor -> eval on non-corridor
    tau_c = X.conformal_tau(sm[ch], ALPHA)
    pc, loc, hic = block_bootstrap_ci((sm[others] > tau_c).astype(float))
    rho_thr = spearmanr(*[a[~(np.isnan(sm) | np.isnan(R["V"]))] for a in (sm, R["V"])]).correlation

    # codebook re-est on Charleston corridor — fit ONLY on ch (calibration-half
    # of the corridor), never on chalf/others, to avoid leaking the evaluation
    # windows into the k-means fit that FPR/rho are then measured against.
    # Standardize the WHOLE Ravenel ride first (matching the raw_ED(rav) call
    # below), then slice the calibration-half prefix from that already
    # z-scored array, so the codebook fit sees the same transform as scoring.
    m.quant.embed.weight.data.copy_(orig_cb)
    ch_sorted = np.sort(ch)      # ch is contiguous ascending window indices by construction
    rav_z = X.zscore(rav)
    ze = CB.encode_ze(m, X.cal_raw_slice(rav_z[ch_sorted[0] * X.STEP:], len(ch_sorted))
                       if len(ch_sorted) else rav_z[:0], standardize=False)
    m.quant.embed.weight.data.copy_(CB.refit_codebook(m, ze))
    E2, D2 = X.raw_ED(m, rav, dev)
    fr2 = X.calibrate(E2, D2, np.isin(np.arange(R["n"]), ch))
    sm2 = X.apply_frozen(E2, D2, fr2)
    tau2 = X.conformal_tau(sm2[ch], ALPHA)
    pcb, locb, hicb = block_bootstrap_ci((sm2[others] > tau2).astype(float))
    rho_cb = spearmanr(*[a[~(np.isnan(sm2) | np.isnan(R["V"]))] for a in (sm2, R["V"])]).correlation
    m.quant.embed.weight.data.copy_(orig_cb)

    print(f"  frozen (Lincoln tau)      FPR {pf:.4f} CI[{lof:.4f},{hif:.4f}]  rho(s_t,VDV)={rho_thr:+.3f}")
    print(f"  conformal (Charleston)    FPR {pc:.4f} CI[{loc:.4f},{hic:.4f}]  rho={rho_thr:+.3f}")
    print(f"  codebook+conformal (ours) FPR {pcb:.4f} CI[{locb:.4f},{hicb:.4f}]  rho={rho_cb:+.3f}")
    pd.DataFrame([
        dict(method="frozen", FPR=round(pf, 4), CI=f"[{lof:.4f},{hif:.4f}]", rho=round(rho_thr, 3)),
        dict(method="conformal", FPR=round(pc, 4), CI=f"[{loc:.4f},{hic:.4f}]", rho=round(rho_thr, 3)),
        dict(method="codebook+conformal", FPR=round(pcb, 4), CI=f"[{locb:.4f},{hicb:.4f}]", rho=round(rho_cb, 3)),
    ]).to_csv(X.OUT / "B_charleston.csv", index=False)
    print(f"\nsaved -> {X.OUT/'B_bootstrap_ci.csv'} , {X.OUT/'B_charleston.csv'}")


if __name__ == "__main__":
    main()
