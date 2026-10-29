#!/usr/bin/env python3
"""
Chainage-aligned drift decomposition on the N St reference corridor.
Plan: refine-logs/EXPERIMENT_PLAN_chainage.md (pre-registered thresholds).

  --geometry   M0 / C000: chainage axis, corridor coverage, windows per bin
  --profiles   M1 / C001-C004: per-bin profiles, cross-campaign Spearman,
               circular-shift null and cross-route null, V4 cross-check

Frozen before any score is computed (plan, Block 0):
  * chainage axis = principal axis of the N Street roadway nodes (OSM),
    origin at the S 11th Street intersection node 134968191
  * corridor = [S 11th 134968191, S 21st 134968247]
  * bins = 20 m from the S 11th node
  * a window is kept if all 65 rows have GPS, its mean chainage lies in the
    corridor, its full row span lies in the corridor, and its complete
    11-window smoothing support is also kept (no truncation at ride ends)
"""
import argparse, math, json
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import aic2_exp as X
import aic2_codebook as CB

OSM = X.OUT / "osm" / "lincoln_nst_osm.xml"
N_S11, N_S21 = "134968191", "134968247"
BIN = 20.0
EDGE = 40.0      # deviation D1: analysis bins need centre >= 40 m from either corridor end
K = X.K_SMOOTH
RIDES = ["autumn", "winter", "spring", "summer", "android"]
LABEL = {"autumn": "Oct 2023", "winter": "Nov-Dec 2023", "spring": "Jan 2024",
         "summer": "May 2024", "android": "Android"}
LAT0, LON0 = 40.8124, -96.6978
NULL_ROUTES = ["13th", "16th", "11th"]          # deviation: Billy dropped (see tracker)


def enu(lat, lon):
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    return np.c_[np.radians(lon - LON0) * 6371000 * math.cos(math.radians(LAT0)),
                 np.radians(lat - LAT0) * 6371000]


def axis():
    r = ET.parse(OSM).getroot()
    nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in r.iter("node")}
    road = []
    for w in r.iter("way"):
        t = {x.get("k"): x.get("v") for x in w.iter("tag")}
        if t.get("name") == "N Street" and t.get("highway") == "tertiary":
            road += [nodes[n.get("ref")] for n in w.iter("nd") if n.get("ref") in nodes]
    P = enu(*np.array(road).T)
    c = P.mean(0); u = np.linalg.svd(P - c)[2][0]
    if u[0] < 0: u = -u                                  # +x = eastward
    o = enu(*nodes[N_S11])[0]
    def ch(Q):
        Q = np.atleast_2d(Q)
        return (Q - o) @ u, (Q - o) @ np.array([-u[1], u[0]])
    e1 = ch(enu(*nodes[N_S21]))[0][0]
    bl = []
    for w in r.iter("way"):
        t = {x.get("k"): x.get("v") for x in w.iter("tag")}
        if t.get("name") == "N Street Bike Lane":
            bl += [nodes[n.get("ref")] for n in w.iter("nd") if n.get("ref") in nodes]
    bs, by = ch(enu(*np.array(bl).T)); o_ = np.argsort(bs)
    axis.bike = lambda q: np.interp(q, bs[o_], by[o_])   # lateral of the bike lane at chainage q
    return ch, 0.0, float(e1)


def load_ride(name):
    if name == "android":
        df = pd.read_csv(X.ANDROID_NST, low_memory=False)
        lat, lon = df["prj_lat"].values, df["prj_lng"].values
        path = X.ANDROID_NST
    else:
        path = str(X.IPHONE / name / "Nst.csv")
        df = pd.read_csv(path, low_memory=False)
        lat = pd.to_numeric(df["Latitude"], errors="coerce").values
        lon = pd.to_numeric(df["Longitude"], errors="coerce").values
    data, _ = X.load_xyz(path)
    return data, lat, lon


def window_geometry(lat, lon, n, ch, e0, e1):
    s, y = ch(enu(lat, lon))
    s[np.isnan(lat) | np.isnan(lon)] = np.nan
    sw = np.full(n, np.nan); yw = np.full(n, np.nan); inside = np.zeros(n, bool)
    for i in range(n):
        seg = s[i * X.STEP:i * X.STEP + X.WIN]
        if np.isnan(seg).any():
            continue
        sw[i] = seg.mean(); yw[i] = np.nanmean(y[i * X.STEP:i * X.STEP + X.WIN]) - axis.bike(seg.mean())
        inside[i] = (seg.min() >= e0) and (seg.max() <= e1)
    keep = np.zeros(n, bool)
    for i in np.where(inside)[0]:
        if i - K >= 0 and i + K < n and inside[i - K:i + K + 1].all():
            keep[i] = True
    return sw, yw, keep


def score(model, data, fr, dev):
    E, D = X.raw_ED(model, data, dev)
    return X.apply_frozen(E, D, fr)


def source_norm(model, dev):
    src, _ = X.load_xyz(str(X.IPHONE / "summer" / "Nst.csv"))
    n0 = CB.windows(src); E0, D0 = X.raw_ED(model, src, dev)
    cal = np.zeros(n0, bool); cal[: n0 // 2] = True
    return X.calibrate(E0, D0, cal)


def analysis_bins(nb, e1):
    c = BIN * (np.arange(nb) + 0.5)
    return (c >= EDGE) & (c <= e1 - EDGE)


def profile(vals, sw, keep, nb):
    idx = np.floor(sw[keep] / BIN).astype(int); v = vals[keep]
    out = np.full(nb, np.nan); cnt = np.zeros(nb, int)
    for b in range(nb):
        m = idx == b
        cnt[b] = m.sum()
        if m.any():
            out[b] = np.nanmedian(v[m])
    return out, cnt


def rho(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return spearmanr(a[ok], b[ok]).correlation if ok.sum() > 5 else np.nan


def run_geometry(args):
    ch, e0, e1 = axis()
    nb = int(np.floor((e1 - e0) / BIN))
    rows, keep_all = [], {}
    for r in RIDES:
        data, lat, lon = load_ride(r)
        n = CB.windows(data)
        sw, yw, keep = window_geometry(lat, lon, n, ch, e0, e1)
        _, cnt = profile(np.ones(n), sw, keep, nb)
        cnt = cnt[analysis_bins(nb, e1)]
        s_valid = sw[~np.isnan(sw)]
        rows.append(dict(ride=r, label=LABEL[r], windows=n, kept=int(keep.sum()),
                         chainage_min=round(float(s_valid.min())), chainage_max=round(float(s_valid.max())),
                         direction="W->E" if s_valid[-1] > s_valid[0] else "E->W",
                         lateral_vs_bikelane_m=round(float(np.nanmedian(yw[keep])), 1),
                         bins=nb, bins_min_windows=int(cnt.min()), bins_median_windows=int(np.median(cnt)),
                         covers_corridor=bool(s_valid.min() <= e0 and s_valid.max() >= e1)))
    df = pd.DataFrame(rows)
    df.to_csv(X.OUT / "CHAIN_geometry.csv", index=False)
    ab = analysis_bins(nb, e1)
    print(f"corridor S11th->S21st: {e1 - e0:.0f} m, {nb} bins of {BIN:.0f} m; analysis bins {ab.sum()} (centres {BIN*(np.where(ab)[0][0]+.5):.0f}-{BIN*(np.where(ab)[0][-1]+.5):.0f} m)")
    print(df.to_string(index=False))
    ok = df.covers_corridor.all() and (df.bins_min_windows >= 5).all() and (df.lateral_vs_bikelane_m.abs() < 8).all()
    print("\nM0 GATE (full coverage, >=5 windows in every analysis bin, |lateral vs bike lane|<8 m):", "PASS" if ok else "FAIL")


def run_profiles(args):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch, e0, e1 = axis(); nb = int(np.floor((e1 - e0) / BIN))
    m = X.A.load_model(X.CKPT, dev)
    fr = source_norm(m, dev)
    S, V, DIR = {}, {}, {}
    for r in RIDES:
        data, lat, lon = load_ride(r)
        n = CB.windows(data)
        sw, _, keep = window_geometry(lat, lon, n, ch, e0, e1)
        st = score(m, data, fr, dev)
        v4 = CB.proxies(data, n)
        S[r], _ = profile(st, sw, keep, nb); V[r], _ = profile(v4, sw, keep, nb)
        S[r][~analysis_bins(nb, e1)] = np.nan; V[r][~analysis_bins(nb, e1)] = np.nan
        s_valid = sw[~np.isnan(sw)]; DIR[r] = "W->E" if s_valid[-1] > s_valid[0] else "E->W"
    pd.DataFrame({**{f"st_{r}": S[r] for r in RIDES}, **{f"v4_{r}": V[r] for r in RIDES}},
                 index=pd.Index(e0 + BIN * (np.arange(nb) + 0.5), name="bin_center_m")
                 ).to_csv(X.OUT / "CHAIN_profiles.csv")

    rng = np.random.default_rng(20261029)
    pairs, nullv = [], []
    for i, a in enumerate(RIDES):
        for b in RIDES[i + 1:]:
            same = DIR[a] == DIR[b]
            rs, rv = rho(S[a], S[b]), rho(V[a], V[b])
            for _ in range(2000 // 10):                 # 200 shifts per pair x 10 pairs = 2000 draws
                k = rng.integers(5, nb - 5)
                nullv.append(rho(S[a], np.roll(S[b], k)))
            pairs.append(dict(pair=f"{a}-{b}", same_direction=same, rho_st=round(rs, 3), rho_v4=round(rv, 3)))
    pdf = pd.DataFrame(pairs)
    q95 = float(np.nanpercentile(nullv, 95))

    # null 2: N St profile vs other routes' own along-track profiles, same campaign
    route_rows = []
    for camp in ["autumn", "winter", "spring", "summer"]:
        for rt in NULL_ROUTES:
            p = X.IPHONE / camp / f"{rt}.csv"
            if not p.exists():
                continue
            data, _ = X.load_xyz(str(p))
            df = pd.read_csv(p, low_memory=False, usecols=["Latitude", "Longitude"])
            P = enu(pd.to_numeric(df.Latitude, errors="coerce"), pd.to_numeric(df.Longitude, errors="coerce"))
            P = pd.DataFrame(P).ffill().bfill().values
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            n = CB.windows(data)
            sw = np.array([d[i * X.STEP:i * X.STEP + X.WIN].mean() for i in range(n)])
            keep = np.zeros(n, bool); keep[K:n - K] = True
            st = score(m, data, fr, dev)
            prof, _ = profile(st, sw, keep, nb)
            route_rows.append(dict(campaign=camp, route=rt, rho_vs_Nst=round(rho(S[camp], prof), 3)))
    rdf = pd.DataFrame(route_rows)

    pdf.to_csv(X.OUT / "CHAIN_pairs.csv", index=False)
    pd.DataFrame(dict(null="circular_shift", draws=len(nullv), q95=q95,
                      cross_route_median=float(rdf.rho_vs_Nst.median()),
                      cross_route_q95=float(rdf.rho_vs_Nst.quantile(.95))), index=[0]
                 ).to_csv(X.OUT / "CHAIN_nulls.csv", index=False)
    print("directions:", DIR)
    print(pdf.to_string(index=False))
    print(f"\ncircular-shift null: 95th pct rho = {q95:.3f} ({len(nullv)} draws)")
    print(f"cross-route null: median rho = {rdf.rho_vs_Nst.median():.3f}, 95th pct = {rdf.rho_vs_Nst.quantile(.95):.3f} ({len(rdf)} rides)")
    same = pdf[pdf.same_direction]; cross = pdf[~pdf.same_direction]
    med_same, med_cross = same.rho_st.median(), cross.rho_st.median()
    med_v4 = same.rho_v4.median()
    print(f"\nsame-direction pairs: median rho_st = {med_same:.3f}, median rho_v4 = {med_v4:.3f}  (n={len(same)})")
    print(f"cross-direction pairs: median rho_st = {med_cross:.3f}, median rho_v4 = {cross.rho_v4.median():.3f}  (n={len(cross)})")
    gate = (med_same >= 0.30) and (med_same > q95) and (med_v4 >= 0.30)
    print("\nM1 GATE (same-dir median rho_st >= 0.30, > shift-null q95, V4 >= 0.30):", "PASS" if gate else "FAIL")
    json.dump(dict(med_same=med_same, med_cross=med_cross, med_v4_same=med_v4, q95_shift=q95,
                   gate="PASS" if gate else "FAIL"), open(X.OUT / "CHAIN_gate_M1.json", "w"), indent=2)




# ============================== M2: decomposition and decision rule ==============================
T_C_MAD, T_E_MAD, MIN_RUN = 1.5, 3.0, 2          # pre-registered (plan, Block 3)


def median_polish(Y, sweeps=20, tol=1e-9):
    """Tukey median polish of a bins x rides matrix (NaN-aware).
    Returns mu, p (per bin), c (per ride), residual matrix."""
    R = Y.copy(); mu = 0.0
    p = np.zeros(Y.shape[0]); c = np.zeros(Y.shape[1])
    for _ in range(sweeps):
        rm = np.nanmedian(R, axis=1); R -= rm[:, None]; p += rm
        m = np.nanmedian(c); c -= m; mu += m
        cm = np.nanmedian(R, axis=0); R -= cm[None, :]; c += cm
        m = np.nanmedian(p); p -= m; mu += m
        if max(np.nanmax(np.abs(rm)), np.nanmax(np.abs(cm))) < tol:
            break
    return mu, p, c, R


def runs(mask, k):
    """Start indices and lengths of runs of True of length >= k."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= k:
                out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def decide(Yh, y):
    """Three-way rule for one new ride profile y against history Yh (bins x rides)."""
    mu, p, c, R = median_polish(Yh)
    c_new = np.nanmedian(y - mu - p)
    eps = y - mu - p - c_new
    T_c = T_C_MAD * np.nanmedian(np.abs(c - np.nanmedian(c))) * 1.4826
    T_e = T_E_MAD * np.nanmedian(np.abs(R[~np.isnan(R)])) * 1.4826
    shift = abs(c_new - np.nanmedian(c)) > T_c
    cands = runs(np.nan_to_num(eps, nan=-np.inf) > T_e, MIN_RUN)
    action = ("recalibrate+inspect" if shift and cands else
              "recalibrate" if shift else "locate-and-inspect" if cands else "ignore")
    return dict(c_new=c_new, c_hist_median=float(np.nanmedian(c)), T_c=T_c, T_e=T_e,
                shift=bool(shift), candidates=cands, action=action, eps=eps)


def run_decompose(args):
    prof = pd.read_csv(X.OUT / "CHAIN_profiles.csv", index_col=0)
    geo = pd.read_csv(X.OUT / "CHAIN_geometry.csv").set_index("ride")
    ab = ~prof[f"st_{RIDES[0]}"].isna()
    Y = prof.loc[ab, [f"st_{r}" for r in RIDES]].values
    centres = prof.index.values[ab]
    mu, p, c, R = median_polish(Y)

    def share(v): return float(np.nanvar(v))
    tot = share(Y - mu)
    rows = dict(var_p=share(p) / tot, var_c=share(np.repeat(c[None, :], len(p), 0)) / tot,
                var_eps=share(R) / tot,
                madsq_p=float((np.nanmedian(np.abs(p)) * 1.4826) ** 2),
                madsq_eps=float((np.nanmedian(np.abs(R)) * 1.4826) ** 2))
    print("pooled median polish (59 bins x 5 rides):")
    print("  c(t):", {LABEL[r]: round(float(v), 4) for r, v in zip(RIDES, c)})
    print("  variance shares  p(s) %.2f   c(t) %.2f   eps %.2f" % (rows["var_p"], rows["var_c"], rows["var_eps"]))
    print("  robust (MAD^2)   p(s) %.5f   eps %.5f   -> p/eps = %.2f" % (rows["madsq_p"], rows["madsq_eps"], rows["madsq_p"] / rows["madsq_eps"]))

    # leave-one-ride-out stability of p(s)
    loo = []
    for j, r in enumerate(RIDES):
        _, pj, _, _ = median_polish(np.delete(Y, j, axis=1))
        loo.append(dict(left_out=r, rho_p_vs_full=round(spearmanr(pj, p).correlation, 3)))
    loo = pd.DataFrame(loo); print(loo.to_string(index=False))

    # per-direction fits (reported, secondary)
    for d in ["E->W", "W->E"]:
        cols = [i for i, r in enumerate(RIDES) if geo.loc[r, "direction"] == d]
        _, pd_, cd_, _ = median_polish(Y[:, cols])
        print(f"  per-direction {d} ({[RIDES[i] for i in cols]}): rho(p_dir, p_pooled) = {spearmanr(pd_, p).correlation:.3f}")

    # consistency with Table 4 (frozen held-out CER, May source)
    tab4 = {"autumn": 0.045, "winter": 0.039, "spring": 0.043, "android": 0.077}
    cc = {r: v for r, v in zip(RIDES, c)}
    rc = spearmanr([cc[k] for k in tab4], list(tab4.values())).correlation
    print(f"  Spearman(c(t), Table-4 frozen CER) over {list(tab4)} = {rc:.3f}  (4 points, descriptive)")

    # LOCO decisions on the real campaigns
    dec = []
    for j, r in enumerate(RIDES):
        out = decide(np.delete(Y, j, axis=1), Y[:, j])
        dec.append(dict(ride=r, label=LABEL[r], c_new=round(out["c_new"], 4), c_hist_median=round(out["c_hist_median"], 4),
                        T_c=round(out["T_c"], 4), shift=out["shift"], T_eps=round(out["T_e"], 4),
                        candidates=";".join(f"{int(centres[i] - BIN/2)}-{int(centres[i] + BIN*(L - .5))}m" for i, L in out["candidates"]) or "-",
                        action=out["action"]))
    dec = pd.DataFrame(dec); print(dec.to_string(index=False))

    pd.DataFrame(dict(bin_center_m=centres, p=p, **{f"eps_{r}": R[:, j] for j, r in enumerate(RIDES)})
                 ).to_csv(X.OUT / "CHAIN_decomp.csv", index=False)
    pd.DataFrame([dict(ride=r, c=float(v)) for r, v in zip(RIDES, c)]).to_csv(X.OUT / "CHAIN_decomp_c.csv", index=False)
    loo.to_csv(X.OUT / "CHAIN_decomp_loo.csv", index=False)
    dec.to_csv(X.OUT / "CHAIN_decisions_loco.csv", index=False)
    gate = (rows["var_p"] > rows["var_eps"]) and (loo.rho_p_vs_full.min() >= 0.8)
    json.dump(dict(**rows, loo_min=float(loo.rho_p_vs_full.min()), rho_c_table4=float(rc),
                   gate="PASS" if gate else "FAIL"), open(X.OUT / "CHAIN_gate_M2.json", "w"), indent=2)
    print("\nM2 GATE (var p > var eps, LOO rho >= 0.8):", "PASS" if gate else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("--geometry", "--profiles", "--decompose"):
        ap.add_argument(f, action="store_true")
    a = ap.parse_args()
    if a.geometry: run_geometry(a)
    if a.profiles: run_profiles(a)
    if a.decompose: run_decompose(a)
