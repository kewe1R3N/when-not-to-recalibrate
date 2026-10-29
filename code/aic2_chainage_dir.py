#!/usr/bin/env python3
"""
Addendum A (refine-logs/PREREG_chainage_addendum_A.md): direction as a design
condition. W->E group = Jan 2024 (spring), May 2024 (summer), Android.

  --g1   within-direction decomposition vs an independent circular-shift null
  --g2   controlled injections on the spring ride (run only if G1 passes)
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import aic2_exp as X
import aic2_codebook as CB
import aic2_chainage as C

WE = ["spring", "summer", "android"]
EW = ["autumn", "winter"]
LOCS = [150, 250, 350, 450, 550, 650, 750, 850, 950, 1050]   # pre-registered
GAINS = [1.2, 1.4, 1.6]
HALF = 12.5


def shares(Y):
    mu, p, c, R = C.median_polish(Y)
    tot = np.nanvar(Y - mu)
    return np.nanvar(p) / tot, np.nanvar(R) / tot, p


def load_matrix(rides):
    prof = pd.read_csv(X.OUT / "CHAIN_profiles.csv", index_col=0)
    ab = ~prof["st_spring"].isna()
    return prof.loc[ab, [f"st_{r}" for r in rides]].values, prof.index.values[ab]


def run_g1():
    Y, centres = load_matrix(WE)
    ps, es, p = shares(Y)
    rng = np.random.default_rng(20261029)
    nb = Y.shape[0]; null = []
    for _ in range(1000):
        Z = np.column_stack([np.roll(Y[:, j], rng.integers(5, nb - 5)) for j in range(Y.shape[1])])
        null.append(shares(Z)[0])
    q95 = float(np.percentile(null, 95))
    loo = []
    for j in range(3):
        _, _, pj = shares(np.delete(Y, j, axis=1))
        loo.append(round(spearmanr(pj, p).correlation, 3))
    Ye, _ = load_matrix(EW); pse, ese, _ = shares(Ye)
    gate = (ps > es) and (ps > q95)
    out = dict(group="W->E", p_share=ps, eps_share=es, null_p_share_q95=q95,
               null_p_share_median=float(np.median(null)), loo_rho=loo,
               EW_descriptive=dict(p_share=pse, eps_share=ese), gate="PASS" if gate else "FAIL")
    json.dump(out, open(X.OUT / "CHAIN_A_G1.json", "w"), indent=2)
    print(f"W->E (3 rides): p-share {ps:.3f}  eps-share {es:.3f}")
    print(f"independent circular-shift null (1000): p-share median {np.median(null):.3f}, q95 {q95:.3f}")
    print(f"LOO (pairs) rho(p_pair, p_full): {loo}")
    print(f"E->W (2 rides, descriptive): p-share {pse:.3f} eps-share {ese:.3f}")
    print("\nA-G1:", out["gate"])
    return gate


def ride_rows_chainage(lat, lon, ch):
    s, _ = ch(C.enu(lat, lon)); return s


def run_g2():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch, e0, e1 = C.axis(); nb = int(np.floor((e1 - e0) / C.BIN)); ab = C.analysis_bins(nb, e1)
    centres = C.BIN * (np.arange(nb) + 0.5)
    m = X.A.load_model(X.CKPT, dev); fr = C.source_norm(m, dev)

    def prof_of(data, lat, lon):
        n = CB.windows(data)
        sw, _, keep = C.window_geometry(lat, lon, n, ch, e0, e1)
        pr, _ = C.profile(C.score(m, data, fr, dev), sw, keep, nb); pr[~ab] = np.nan
        return pr[ab]

    raw = {r: C.load_ride(r) for r in WE}
    base = {r: prof_of(*raw[r]) for r in WE}
    cen = centres[ab]

    def history(rides): return np.column_stack([base[r] for r in rides])

    def bins_of(cands): return {i + k for i, L in cands for k in range(L)}

    H = history(["summer", "android"])
    base_dec = C.decide(H, base["spring"]); base_bins = bins_of(base_dec["candidates"])
    data0, lat0, lon0 = raw["spring"]; s_rows = ride_rows_chainage(lat0, lon0, ch)

    def inject(g_local=None, s0=None, g_uniform=None):
        x = data0.copy()
        if g_uniform is not None:
            mu = x.mean(0); x = mu + g_uniform * (x - mu)
        if g_local is not None:
            rm = np.abs(s_rows - s0) <= HALF
            mu = x[rm].mean(0); x[rm] = mu + g_local * (x[rm] - mu)
        return x.astype(np.float32)

    rows = []
    def record(case, g, s0, y):
        d = C.decide(H, y); b = bins_of(d["candidates"])
        hit = None
        if s0 is not None:
            tgt = int(np.argmin(np.abs(cen - s0)))
            hit = any(abs(k - tgt) <= 1 for k in b)
        new_bins = sorted(b - base_bins)
        rows.append(dict(case=case, g=g, s0=s0, shift=d["shift"], action=d["action"],
                         c_new=round(d["c_new"], 4), T_c=round(d["T_c"], 4),
                         candidate_bins_m=";".join(str(int(cen[k])) for k in sorted(b)) or "-",
                         new_vs_baseline=len(new_bins), hit=hit))

    record("baseline (un-injected spring)", 1.0, None, base["spring"])
    for g in [1.0] + GAINS:
        for s0 in LOCS:
            record("placebo" if g == 1.0 else "localized", g, s0, prof_of(inject(g, s0), lat0, lon0))
    record("uniform", 1.3, None, prof_of(inject(g_uniform=1.3), lat0, lon0))
    record("both", 1.6, 550, prof_of(inject(1.6, 550, 1.3), lat0, lon0))
    d = C.decide(history(["spring", "summer"]), base["android"])
    rows.append(dict(case="real: Android vs Jan+May", g=None, s0=None, shift=d["shift"], action=d["action"],
                     c_new=round(d["c_new"], 4), T_c=round(d["T_c"], 4),
                     candidate_bins_m=";".join(str(int(cen[i + k])) for i, L in d["candidates"] for k in range(L)) or "-",
                     new_vs_baseline=None, hit=None))
    df = pd.DataFrame(rows); df.to_csv(X.OUT / "CHAIN_A_G2.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 60):
        print(df.to_string(index=False))

    loc = df[df.case == "localized"]
    det = {g: int(loc[loc.g == g].hit.sum()) for g in GAINS}
    pl = df[df.case == "placebo"]; fp = int(pl.new_vs_baseline.sum()); fp_rate = fp / (10 * ab.sum())
    uni = df[df.case == "uniform"].iloc[0]; both = df[df.case == "both"].iloc[0]
    both_hit = any(abs(int(v) - 550) <= 20 for v in str(both.candidate_bins_m).split(";") if v != "-")
    crit = dict(det_16=bool(det[1.6] >= 8), placebo=bool(fp_rate <= 0.05),
                uniform=bool(uni["shift"] and uni.new_vs_baseline <= 1), both=bool(both["shift"] and both_hit))
    print(f"\ndetection (of 10, within +-1 bin): {det}")
    print(f"placebo new candidate bins: {fp} / {10*ab.sum()} = {fp_rate:.3f}")
    print(f"criteria: {crit}")
    print("A-G2:", "PASS" if all(crit.values()) else "FAIL")
    json.dump(dict(detection={str(k): int(v) for k, v in det.items()}, placebo_rate=float(fp_rate), criteria=crit,
                   gate="PASS" if all(crit.values()) else "FAIL"), open(X.OUT / "CHAIN_A_G2.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--g1", action="store_true"); ap.add_argument("--g2", action="store_true")
    a = ap.parse_args()
    if a.g1: run_g1()
    if a.g2:
        if json.load(open(X.OUT / "CHAIN_A_G1.json"))["gate"] != "PASS":
            raise SystemExit("A-G1 did not pass; A-G2 is not run (pre-registered).")
        run_g2()
