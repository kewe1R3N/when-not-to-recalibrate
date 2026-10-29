#!/usr/bin/env python3
"""
DET-v2 — strengthen the deterioration negative control by matching the DiD
reference to the target's surface/behavior.

Hypothesis: the single global asphalt corridor (N St) absorbs only ~21% of the
cross-campaign nuisance variance because it does not represent the seasonal
response of heterogeneous routes/surfaces (parallel-trends violation). A
reference that shares the route, surface, and rider-behavior distribution should
absorb far more.

Estimators compared (variance absorption of cross-campaign change on routes
believed ~stable over one freeze-thaw year):
  (E0) Global-corridor DiD : control = N St whole-ride delta (the v1 estimator).
  (E1) Within-route DiD    : control = the SAME route's smooth substrate
        (bottom-tercile-VDV windows); target = its rough (top-tercile) windows.
        Surface/behavior matched by construction; assumption becomes "season
        affects a route's smooth and rough windows equally".
  (E2) Pooled-smooth DiD   : control = smooth windows pooled across ALL routes.

We also re-run the semi-synthetic positive control under E1 (injected loc is a
rough location; the route's smooth substrate is the local control).
"""
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
import aic2_exp as X

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMPS = ["autumn", "winter", "spring", "summer"]
CDATE = {"autumn": "Oct 2023", "winter": "Nov-Dec 2023",
         "spring": "Jan 2024", "summer": "May 2024"}
ROUTES = ["11th", "13th", "16th", "billy"]
CORR = "Nst"


def load_xyz_gps(camp, route):
    df = pd.read_csv(str(X.IPHONE / camp / f"{route}.csv"), low_memory=False)
    xyz = df[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    la = pd.to_numeric(df.get("Latitude"), errors="coerce").values
    lo = pd.to_numeric(df.get("Longitude"), errors="coerce").values
    return xyz, la, lo


def cond_and_vdv(xyz, fr):
    n = (len(xyz) - X.WIN)//X.STEP + 1
    E, D = X.raw_ED(X.MODEL, xyz, dev)
    C = X.apply_frozen(E, D, fr)
    V = np.array([X.A.compute_window_proxies(xyz[i*X.STEP:i*X.STEP+X.WIN, 0],
                                             xyz[i*X.STEP:i*X.STEP+X.WIN, 1],
                                             xyz[i*X.STEP:i*X.STEP+X.WIN, 2],
                                             X.FS, X.A.config.HP_CUTOFF)["VDV_w"]
                 for i in range(n)])
    return C, V


def main():
    X.MODEL = X.A.load_model(X.CKPT, dev)
    cx, _, _ = load_xyz_gps("summer", CORR)
    nS = (len(cx) - X.WIN)//X.STEP + 1
    calm = np.zeros(nS, bool); calm[:nS//2] = True
    E0, D0 = X.raw_ED(X.MODEL, cx, dev)
    fr = X.calibrate(E0, D0, calm)

    # precompute condition + VDV for corridor and all routes x campaigns
    store = {}
    for camp in CAMPS:
        for r in [CORR] + ROUTES:
            try:
                xyz, _, _ = load_xyz_gps(camp, r)
            except FileNotFoundError:
                continue
            C, V = cond_and_vdv(xyz, fr)
            ok = ~(np.isnan(C) | np.isnan(V))
            store[(camp, r)] = (C[ok], V[ok])

    def smooth_rough(camp, r):
        C, V = store[(camp, r)]
        lo, hi = np.nanpercentile(V, 33), np.nanpercentile(V, 67)
        return C[V <= lo], C[V >= hi]      # smooth (stable), rough (defect-prone)

    corrC = {c: float(np.nanmedian(store[(c, CORR)][0])) for c in CAMPS}
    pairs = [(CAMPS[i], CAMPS[j]) for i in range(len(CAMPS)) for j in range(len(CAMPS)) if i < j]

    # pooled smooth control per campaign
    pooledS = {}
    for c in CAMPS:
        vals = []
        for r in ROUTES:
            if (c, r) in store:
                s, _ = smooth_rough(c, r); vals.append(s)
        pooledS[c] = float(np.nanmedian(np.concatenate(vals)))

    raw, e0, e1, e2 = [], [], [], []
    rows = []
    for (t0, t1) in pairs:
        dcorr = corrC[t1] - corrC[t0]
        dpool = pooledS[t1] - pooledS[t0]
        for r in ROUTES:
            if (t0, r) not in store or (t1, r) not in store:
                continue
            s0, h0 = smooth_rough(t0, r); s1, h1 = smooth_rough(t1, r)
            rough0, rough1 = np.nanmedian(h0), np.nanmedian(h1)
            smooth0, smooth1 = np.nanmedian(s0), np.nanmedian(s1)
            raw_d = rough1 - rough0                         # apparent change (rough)
            d_e0 = raw_d - dcorr                            # global corridor
            d_e1 = raw_d - (smooth1 - smooth0)              # within-route smooth
            d_e2 = raw_d - dpool                            # pooled smooth
            raw.append(raw_d); e0.append(d_e0); e1.append(d_e1); e2.append(d_e2)
            rows.append(dict(pair=f"{CDATE[t0]}->{CDATE[t1]}", route=r,
                             raw=round(raw_d, 4), E0_global=round(d_e0, 4),
                             E1_within=round(d_e1, 4), E2_pooled=round(d_e2, 4)))
    raw, e0, e1, e2 = map(np.array, (raw, e0, e1, e2))
    pd.DataFrame(rows).to_csv(X.OUT / "DET_v2_estimators.csv", index=False)

    def absorb(res):
        return 100 * (1 - np.var(res)/np.var(raw))
    print("=== variance absorption of cross-campaign change (routes ~stable) ===")
    print(f"  raw |delta| mean={np.mean(np.abs(raw)):.4f}  var={np.var(raw):.5f}")
    for name, res in [("E0 global corridor", e0), ("E1 within-route smooth", e1),
                      ("E2 pooled smooth", e2)]:
        print(f"  {name:24s}: |D| mean={np.mean(np.abs(res)):.4f}  "
              f"var={np.var(res):.5f}  absorbed={absorb(res):+.1f}%")

    # ===== positive control under E1 (within-route) =====
    print("\n=== (POS) semi-synthetic recovery under E1 within-route DiD (13th) ===")
    inj_factor = {"autumn": 1.0, "winter": 1.2, "spring": 1.4, "summer": 1.6}
    wx, wla, wlo = load_xyz_gps("winter", "13th")
    nW = (len(wx)-X.WIN)//X.STEP + 1
    la = np.array([np.nanmean(wla[i*X.STEP:i*X.STEP+X.WIN]) for i in range(nW)])
    lo = np.array([np.nanmean(wlo[i*X.STEP:i*X.STEP+X.WIN]) for i in range(nW)])
    ok = ~(np.isnan(la) | np.isnan(lo))
    loc_lat, loc_lon = np.nanmedian(la[ok]), np.nanmedian(lo[ok])
    RAD = 0.00025
    prows = []
    for camp in CAMPS:
        xyz, laR, loR = load_xyz_gps(camp, "13th")
        n = (len(xyz)-X.WIN)//X.STEP + 1
        laW = np.array([np.nanmean(laR[i*X.STEP:i*X.STEP+X.WIN]) for i in range(n)])
        loW = np.array([np.nanmean(loR[i*X.STEP:i*X.STEP+X.WIN]) for i in range(n)])
        inj = (np.abs(laW-loc_lat) < RAD) & (np.abs(loW-loc_lon) < RAD)
        g = inj_factor[camp]
        rowmask = np.zeros(len(xyz), bool)
        for i in np.where(inj)[0]:
            rowmask[i*X.STEP:i*X.STEP+X.WIN] = True
        xj = xyz.copy()
        mu = xj[rowmask].mean(0)
        xj[rowmask] = mu + g*(xj[rowmask]-mu)
        Cj, Vj = cond_and_vdv(xj, fr)
        okc = ~(np.isnan(Cj) | np.isnan(Vj))
        # within-route smooth reference on the (injected) ride
        loV = np.nanpercentile(Vj[okc], 33)
        smoothref = float(np.nanmedian(Cj[okc & (Vj <= loV)]))
        injloc = float(np.nanmedian(Cj[inj])) if inj.sum() else np.nan
        prows.append(dict(camp=CDATE[camp], factor=g, C_injloc=injloc,
                          smooth_ref=smoothref))
    pdf = pd.DataFrame(prows)
    b = pdf.iloc[0]
    pdf["D_within"] = (pdf.C_injloc - b.C_injloc) - (pdf.smooth_ref - b.smooth_ref)
    pdf.to_csv(X.OUT / "DET_v2_synthetic.csv", index=False)
    print(pdf.round(4).to_string(index=False))
    print(f"  Spearman(factor, D_within@injected) = "
          f"{spearmanr(pdf.factor, pdf.D_within).correlation:+.3f}")
    print("saved -> DET_v2_estimators.csv, DET_v2_synthetic.csv")


if __name__ == "__main__":
    main()
