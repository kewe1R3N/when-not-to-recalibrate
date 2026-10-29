#!/usr/bin/env python3
"""
CITY — Tier-0 Charleston city axis from data already on disk.

Datasets (verified 2026-07-07):
  Citadel iPhone : data_process_Citadel_iphone2/filtered1..20.csv — 10 Hz exactly
                   (10 rows per GPS fix), g units, gravity present but TILTED
                   (mean gravity ~ (0.03, -0.74, -0.67)): a real-world rotation
                   case. One long tour chopped into 20 segments; segments 6&16
                   share ~56 GPS cells (~600 m street ridden twice) and 3&12
                   share ~25 — genuine REVISIT corridors.
  Ravenel Android: full Sensor Logger dump, 105 Hz, m/s^2, gravity on Y;
                   has a Gravity stream -> principled per-sample reorientation.

Lanes:
  L1 ranking   : per-segment rho(s_t, VDV) after tilt harmonization (20 cells)
                 + Ravenel compound case.
  L2 calibration: revisit corridor — calibrate on pass A (seg 6), evaluate FPR
                 on pass B (seg 16), both directions; frozen-Lincoln reference
                 (city drift) vs conformal recal on pass A (the cure). Repl. 3&12.
  L3 trigger   : codebook-usage JS of Citadel/Ravenel vs Lincoln summer ref ->
                 extends the environmental JS landscape with a 'city' level.

Harmonization = MINIMAL ROTATION taking the segment's mean gravity direction to
(0,0,-1) (Lincoln convention: gravity on -Z in g units). Yaw is left free —
justified empirically by the fault grid (in-plane permutations barely move
ranking) and stated as a caveat.
"""
import glob
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from scipy.signal import resample_poly
import aic2_exp as X
import aic2_codebook as CB
from aic2_fault_injection import code_indices, usage_hist, js_div

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = X.ALPHA
IP = sorted(glob.glob(str(X.CODE / "data_process_Citadel_iphone2" / "filtered*.csv")),
            key=lambda p: int("".join(filter(str.isdigit, p.split("/")[-1]))))
RAV_DIR = X.CODE / "code_version_sep18" / "New_Android_Ravenel_Bridge"


def rot_to_minus_z(gvec):
    """Minimal rotation matrix taking unit vector gvec -> (0,0,-1) (Rodrigues)."""
    g = gvec / (np.linalg.norm(gvec) + 1e-12)
    t = np.array([0.0, 0.0, -1.0])
    v = np.cross(g, t); s = np.linalg.norm(v); c = float(np.dot(g, t))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def load_citadel_segment(path):
    df = pd.read_csv(path, low_memory=False)
    xyz = df[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").fillna(0).values
    R = rot_to_minus_z(xyz.mean(axis=0))          # tilt harmonization
    return (xyz @ R.T).astype(np.float32), df


def seg_rho(m, data):
    n = CB.windows(data)
    if n < 30:
        return None, n
    E, D = X.raw_ED(m, data, dev)
    st = CB.st_selfnorm(E, D)
    V = CB.proxies(data, n)
    ok = ~(np.isnan(st) | np.isnan(V))
    return float(spearmanr(st[ok], V[ok]).correlation), int(ok.sum())


def window_mask_from_rows(row_mask, n):
    """Windows whose full [i*STEP, i*STEP+WIN) span lies in row_mask."""
    out = np.zeros(n, bool)
    for i in range(n):
        s = i * X.STEP
        out[i] = bool(row_mask[s:s + X.WIN].all())
    return out


def trim_to_full_smoothing_support(mask, k=None):
    """Shrink a window mask to windows whose COMPLETE centered rolling-median
    smoothing neighborhood (radius k, default X.K_SMOOTH) is also in the mask.
    Without this, apply_frozen()'s smoothing (run over the whole ride before
    any corridor mask is applied) lets a "corridor" window's returned score
    quietly depend on neighboring non-corridor windows -- exactly the leak
    the window-span check in window_mask_from_rows does not catch."""
    if k is None:
        k = X.K_SMOOTH
    n = len(mask)
    out = mask.copy()
    idx = np.where(mask)[0]
    for i in idx:
        lo, hi = max(0, i - k), min(n, i + k + 1)
        if not mask[lo:hi].all():
            out[i] = False
    return out


def main():
    m = X.A.load_model(X.CKPT, dev)
    K = m.quant.embed.weight.shape[0]

    # Lincoln summer reference (frozen thresholds + JS reference), as in M1/FI
    src_data, _ = X.load_xyz(str(X.IPHONE / "summer" / "Nst.csv"))
    n0 = CB.windows(src_data)
    E0, D0 = X.raw_ED(m, src_data, dev)
    cal0 = np.zeros(n0, bool); cal0[: n0 // 2] = True
    fr_lin = X.calibrate(E0, D0, cal0)
    p_src = usage_hist(code_indices(m, src_data[: len(src_data) // 2]), K)

    # ================= L1: Citadel iPhone ranking (20 segments) =================
    print("=== L1: Citadel iPhone per-segment rho(s_t, VDV), tilt-harmonized ===")
    rows, seg_cache = [], {}
    for i, p in enumerate(IP, 1):
        data, df = load_citadel_segment(p)
        seg_cache[i] = (data, df)
        r, n = seg_rho(m, data)
        rows.append(dict(segment=i, n=n, rho=None if r is None else round(r, 3)))
        if r is not None:
            print(f"  seg{i:2d}: n={n:4d}  rho={r:+.3f}")
    df1 = pd.DataFrame(rows)
    df1.to_csv(X.OUT / "CITY_citadel_ranking.csv", index=False)
    val = df1.rho.dropna()
    print(f"  -> {len(val)} segments, rho range {val.min():+.3f}..{val.max():+.3f}, "
          f"median {val.median():+.3f}")

    # ================= L2: revisit corridors =================
    print("\n=== L2: revisit-corridor calibration (cross-pass) ===")
    ci_rows = []
    for segA, segB in [(6, 16), (3, 12)]:
        dA, fA = seg_cache[segA]; dB, fB = seg_cache[segB]
        cellsA = list(zip(fA["Latitude"].round(4), fA["Longitude"].round(4)))
        cellsB = list(zip(fB["Latitude"].round(4), fB["Longitude"].round(4)))
        shared = set(cellsA) & set(cellsB)
        mA = np.array([c in shared for c in cellsA])
        mB = np.array([c in shared for c in cellsB])
        EA, DA = X.raw_ED(m, dA, dev); EB, DB = X.raw_ED(m, dB, dev)
        wA = trim_to_full_smoothing_support(window_mask_from_rows(mA, CB.windows(dA)))
        wB = trim_to_full_smoothing_support(window_mask_from_rows(mB, CB.windows(dB)))
        if wA.sum() < 30 or wB.sum() < 30:
            print(f"  ({segA},{segB}): too few corridor windows "
                  f"({wA.sum()},{wB.sum()}) — skipped"); continue
        for (tagA, Ea, Da, wa, tagB, Eb, Db, wb) in [
                (segA, EA, DA, wA, segB, EB, DB, wB),
                (segB, EB, DB, wB, segA, EA, DA, wA)]:
            # frozen Lincoln thresholds on pass B corridor (city drift)
            smB_lin = X.apply_frozen(Eb, Db, fr_lin)
            fpr_frozen = float(np.mean(smB_lin[wb] > fr_lin["tau"]))
            # conformal recal on pass A -> evaluate pass B (Lincoln normalization)
            smA_lin = X.apply_frozen(Ea, Da, fr_lin)
            tau_c = X.conformal_tau(smA_lin[wa], ALPHA)
            fpr_conf = float(np.mean(smB_lin[wb] > tau_c))
            ci_rows.append(dict(cal_pass=f"seg{tagA}", eval_pass=f"seg{tagB}",
                                n_cal=int(wa.sum()), n_eval=int(wb.sum()),
                                FPR_frozen_lincoln=round(fpr_frozen, 4),
                                FPR_conformal_revisit=round(fpr_conf, 4)))
            print(f"  cal seg{tagA} -> eval seg{tagB}: n={wa.sum()}/{wb.sum()}  "
                  f"frozen-Lincoln FPR {fpr_frozen:.4f} ({fpr_frozen/ALPHA:.1f}x)  "
                  f"conformal-revisit FPR {fpr_conf:.4f} ({fpr_conf/ALPHA:.1f}x)")
    pd.DataFrame(ci_rows).to_csv(X.OUT / "CITY_revisit_corridor.csv", index=False)

    # ================= L3: JS landscape city rows =================
    print("\n=== L3: codebook-usage JS vs Lincoln summer reference ===")
    js_rows = []
    cit_all = np.concatenate([seg_cache[i][0] for i in sorted(seg_cache)][:10])
    p_cit = usage_hist(code_indices(m, cit_all), K)
    js_cit = js_div(p_cit, p_src)
    js_rows.append(dict(context="citadel_iphone(city)", kind="city", JS=round(js_cit, 4)))
    print(f"  citadel iPhone (city)          JS = {js_cit:.4f}")

    # ================= Ravenel: gravity-stream reorientation =================
    print("\n=== Ravenel: per-sample gravity reorientation + ranking ===")
    acc = pd.read_csv(RAV_DIR / "TotalAcceleration.csv")
    grav = pd.read_csv(RAV_DIR / "Gravity.csv")
    a = acc[["x", "y", "z"]].values
    g = grav[["x", "y", "z"]].values
    n = min(len(a), len(g)); a, g = a[:n], g[:n]
    out = np.empty_like(a)
    step = 1024
    for s in range(0, n, step):                      # chunked per-sample rotation
        for j in range(s, min(s + step, n)):
            out[j] = rot_to_minus_z(g[j]) @ a[j]
    out /= 9.81                                       # m/s^2 -> g
    rate = n / (acc["seconds_elapsed"].iloc[n - 1] - acc["seconds_elapsed"].iloc[0])
    rav10 = resample_poly(out, up=2, down=21, axis=0).astype(np.float32)
    print(f"  rate={rate:.1f} Hz -> resampled {len(rav10)} samples @10 Hz; "
          f"mean after reorient: {out.mean(0).round(3)}")
    r, nn = seg_rho(m, rav10)
    print(f"  Ravenel ranking rho(s_t,VDV) = {r:+.3f}  (n={nn})")
    p_rav = usage_hist(code_indices(m, rav10), K)
    js_rav = js_div(p_rav, p_src)
    js_rows.append(dict(context="ravenel(city+device+app)", kind="city_compound",
                        JS=round(js_rav, 4)))
    print(f"  Ravenel JS = {js_rav:.4f}")

    jc = pd.read_csv(X.OUT / "FI_js_context.csv")
    new_ctx = {r["context"] for r in js_rows}
    jc = jc[~jc["context"].isin(new_ctx)]                    # drop stale rows before re-appending
    jc = pd.concat([jc, pd.DataFrame(js_rows)], ignore_index=True)
    jc.to_csv(X.OUT / "FI_js_context.csv", index=False)
    pd.DataFrame([dict(ravenel_rho=r, ravenel_n=nn, ravenel_js=js_rav)]).to_csv(
        X.OUT / "CITY_ravenel.csv", index=False)
    print(f"\nsaved -> CITY_citadel_ranking.csv, CITY_revisit_corridor.csv, "
          f"CITY_ravenel.csv, FI_js_context.csv (+city rows)")


if __name__ == "__main__":
    main()
