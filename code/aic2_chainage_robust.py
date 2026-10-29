#!/usr/bin/env python3
"""
Robustness checks requested by the independent review of the chainage analysis
(recorded as deviations / sensitivity analyses in EXPERIMENT_TRACKER_chainage.md):

  R1  metric reconciliation: the plan specified MAD-based shares; M2 and A-G1
      were gated on ordinary variance ratios. Report both, for M2 and A-G1,
      each against the matching null.
  R2  matched-statistic null for M1: statistic = median Spearman over the
      same-direction (resp. cross-direction) pairs on the same 59 analysis
      bins; null = independent circular shifts of each ride's profile WITHIN
      those 59 bins, same statistic. 2000 draws. Learned score and V4.
  R3  source-excluded normalization: refit the frozen normalization on a ride
      outside every comparison (May 2024, 16th St, first half) and repeat
      M1 (R2 statistics) and the W->E decomposition.
"""
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import aic2_exp as X
import aic2_codebook as CB
import aic2_chainage as C

RIDES = C.RIDES
DIRN = {"autumn": "E->W", "winter": "E->W", "spring": "W->E", "summer": "W->E", "android": "W->E"}
WE = [2, 3, 4]
rng = np.random.default_rng(20261030)


def rho(a, b):
    ok = ~(np.isnan(a) | np.isnan(b)); return spearmanr(a[ok], b[ok]).correlation


def pair_stats(Y):
    same, cross = [], []
    for i in range(5):
        for j in range(i + 1, 5):
            (same if DIRN[RIDES[i]] == DIRN[RIDES[j]] else cross).append(rho(Y[:, i], Y[:, j]))
    return np.median(same), np.median(cross)


def matched_null(Y, draws=2000):
    nb = Y.shape[0]; S, Cr = [], []
    for _ in range(draws):
        Z = np.column_stack([np.roll(Y[:, j], rng.integers(5, nb - 5)) for j in range(Y.shape[1])])
        s, c = pair_stats(Z); S.append(s); Cr.append(c)
    return np.percentile(S, 95), np.percentile(Cr, 95), np.median(S)


def shares(Y):
    mu, p, c, R = C.median_polish(Y)
    tot = np.nanvar(Y - mu)
    mad = lambda v: (np.nanmedian(np.abs(v - np.nanmedian(v))) * 1.4826) ** 2
    return dict(var_p=np.nanvar(p) / tot, var_eps=np.nanvar(R) / tot,
                mad_p=mad(p), mad_eps=mad(R[~np.isnan(R)]), mad_ratio=mad(p) / mad(R[~np.isnan(R)]))


def decomp_with_null(Y, draws=1000):
    obs = shares(Y); nb = Y.shape[0]; nv, nm = [], []
    for _ in range(draws):
        Z = np.column_stack([np.roll(Y[:, j], rng.integers(5, nb - 5)) for j in range(Y.shape[1])])
        s = shares(Z); nv.append(s["var_p"]); nm.append(s["mad_ratio"])
    return obs, np.percentile(nv, 95), np.percentile(nm, 95), np.median(nv), np.median(nm)


def profiles(fr, m, dev, ch, e0, e1, nb, ab):
    S, V = [], []
    for r in RIDES:
        data, lat, lon = C.load_ride(r); n = CB.windows(data)
        sw, _, keep = C.window_geometry(lat, lon, n, ch, e0, e1)
        s, _ = C.profile(C.score(m, data, fr, dev), sw, keep, nb)
        v, _ = C.profile(CB.proxies(data, n), sw, keep, nb)
        S.append(s[ab]); V.append(v[ab])
    return np.column_stack(S), np.column_stack(V)


def report(tag, Y, V, out):
    ms, mc = pair_stats(Y); q_s, q_c, med0 = matched_null(Y)
    vs, vc = pair_stats(V); vq_s, vq_c, _ = matched_null(V)
    out[tag + "_M1"] = dict(same_median=ms, same_null_q95=q_s, same_null_median=med0, cross_median=mc, cross_null_q95=q_c,
                           v4_same_median=vs, v4_same_null_q95=vq_s, v4_cross_median=vc, v4_cross_null_q95=vq_c,
                           pass_same=bool(ms >= 0.30 and ms > q_s and vs >= 0.30 and vs > vq_s))
    print(f"[{tag}] M1 matched null: same-dir median rho {ms:.3f} (null q95 {q_s:.3f}, null median {med0:.3f}); "
          f"cross-dir {mc:.3f} (null q95 {q_c:.3f})")
    print(f"[{tag}]                 V4 same-dir {vs:.3f} (null q95 {vq_s:.3f}); cross-dir {vc:.3f} (null q95 {vq_c:.3f})")
    for name, cols in [("pooled-5 (M2)", list(range(5))), ("W->E (A-G1)", WE)]:
        obs, qv, qm, mv, mm = decomp_with_null(Y[:, cols])
        out[f"{tag}_{name}"] = dict(**obs, null_var_p_q95=qv, null_var_p_median=mv, null_mad_ratio_q95=qm, null_mad_ratio_median=mm)
        print(f"[{tag}] {name:14s} var-share p {obs['var_p']:.3f} (null q95 {qv:.3f}, median {mv:.3f}) | eps {obs['var_eps']:.3f}"
              f" || MAD^2 ratio p/eps {obs['mad_ratio']:.2f} (null q95 {qm:.2f}, median {mm:.2f})")


def main():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch, e0, e1 = C.axis(); nb = int(np.floor((e1 - e0) / C.BIN)); ab = C.analysis_bins(nb, e1)
    m = X.A.load_model(X.CKPT, dev)
    out = {}
    # original normalization (May N St, first half) -- as in M1/M2/A-G1
    fr0 = C.source_norm(m, dev)
    Y0, V0 = profiles(fr0, m, dev, ch, e0, e1, nb, ab)
    report("source=May NSt", Y0, V0, out)
    # source-excluded normalization (May 16th St, first half): outside every comparison
    src, _ = X.load_xyz(str(X.IPHONE / "summer" / "16th.csv"))
    n0 = CB.windows(src); E, D = X.raw_ED(m, src, dev)
    cal = np.zeros(n0, bool); cal[: n0 // 2] = True
    fr1 = X.calibrate(E, D, cal)
    Y1, V1 = profiles(fr1, m, dev, ch, e0, e1, nb, ab)
    report("source=May 16th (excluded)", Y1, V1, out)
    print(f"\nprofile agreement between normalizations: rho per ride = "
          f"{[round(rho(Y0[:, j], Y1[:, j]), 3) for j in range(5)]}")
    json.dump(out, open(X.OUT / "CHAIN_robustness.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
