#!/usr/bin/env python3
"""
DET — reversible-shift vs irreversible-deterioration disentanglement via a
corridor difference-in-differences (DiD) estimator.

Idea: the reference corridor (N St) is assumed physically stable over the study
period, so any cross-campaign change in its condition index is REVERSIBLE
nuisance (season/device). Treating the corridor as the DiD control arm, the
route-specific residual change is the IRREVERSIBLE deterioration estimate:

    D(r; t0->t1) = [C(r,t1) - C(r,t0)]  -  [C(corridor,t1) - C(corridor,t0)]

C(.) is the frozen-normalized smoothed anomaly score (Paper-1's CDI condition
index) on a COMMON scale (fixed normalization from the summer N St corridor) so
that cross-campaign levels are comparable (no per-ride self-normalization).

Two controlled validations (no deterioration ground truth exists):
  (NEG) Variance absorption: over all routes x campaign-pairs, does the corridor
        absorb the common seasonal component? Compare spread of the raw
        cross-campaign delta vs the DiD residual; the residual should be small
        and centered near 0 if routes are ~stable over one freeze-thaw year.
  (POS) Semi-synthetic recovery: inject a progressive defect (growing roughness)
        into a fixed GPS location of one route across the chronologically ordered
        campaigns; the DiD estimate at the injected location should track the
        injected magnitude while control locations stay flat.

Parallel-trends assumption (stated, and probed): the nuisance time effect is
common across routes. We probe it by the agreement of per-route seasonal deltas.
"""
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
import aic2_exp as X

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMPS = ["autumn", "winter", "spring", "summer"]        # chronological order
CDATE = {"autumn": "Oct 2023", "winter": "Nov-Dec 2023",
         "spring": "Jan 2024", "summer": "May 2024"}
ROUTES = ["11th", "13th", "16th", "billy"]              # non-corridor routes
CORR = "Nst"


def load_route(camp, route):
    df = pd.read_csv(str(X.IPHONE / camp / f"{route}.csv"), low_memory=False)
    xyz = df[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    lat = pd.to_numeric(df.get("Latitude"), errors="coerce").values
    lon = pd.to_numeric(df.get("Longitude"), errors="coerce").values
    return xyz, lat, lon


def cond_index(xyz, fr):
    """Frozen-normalized smoothed condition index C per window (common scale)."""
    E, D = X.raw_ED(X.MODEL, xyz, dev)
    return X.apply_frozen(E, D, fr)


def win_gps(lat, lon, n):
    la = np.array([np.nanmean(lat[i*X.STEP:i*X.STEP+X.WIN]) for i in range(n)])
    lo = np.array([np.nanmean(lon[i*X.STEP:i*X.STEP+X.WIN]) for i in range(n)])
    return la, lo


def main():
    X.MODEL = X.A.load_model(X.CKPT, dev)
    # common scale: freeze normalization on summer N St corridor (first half)
    cx, _, _ = load_route("summer", CORR)
    nS = (len(cx) - X.WIN)//X.STEP + 1
    calm = np.zeros(nS, bool); calm[:nS//2] = True
    E0, D0 = X.raw_ED(X.MODEL, cx, dev)
    fr = X.calibrate(E0, D0, calm)

    # ---- condition index per route x campaign (median over ride) ----
    C = {}   # C[(camp,route)] = median condition index
    corridorC = {}
    for camp in CAMPS:
        cx, _, _ = load_route(camp, CORR)
        corridorC[camp] = float(np.nanmedian(cond_index(cx, fr)))
        for r in ROUTES:
            try:
                xyz, _, _ = load_route(camp, r)
            except FileNotFoundError:
                continue
            C[(camp, r)] = float(np.nanmedian(cond_index(xyz, fr)))

    # ===== (NEG) variance absorption over routes x adjacent campaign pairs =====
    print("=== (NEG) Corridor-DiD variance absorption (condition index) ===")
    raw_deltas, did_resid, rows = [], [], []
    pairs = [(CAMPS[i], CAMPS[j]) for i in range(len(CAMPS)) for j in range(len(CAMPS)) if i < j]
    for (t0, t1) in pairs:
        dcorr = corridorC[t1] - corridorC[t0]
        for r in ROUTES:
            if (t0, r) not in C or (t1, r) not in C:
                continue
            raw = C[(t1, r)] - C[(t0, r)]
            did = raw - dcorr
            raw_deltas.append(raw); did_resid.append(did)
            rows.append(dict(pair=f"{CDATE[t0]}->{CDATE[t1]}", route=r,
                             raw_delta=round(raw, 4), corridor_delta=round(dcorr, 4),
                             DiD_deterioration=round(did, 4)))
    raw_deltas = np.array(raw_deltas); did_resid = np.array(did_resid)
    df = pd.DataFrame(rows); df.to_csv(X.OUT / "DET_did_variance.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n  raw cross-campaign |delta|: mean={np.mean(np.abs(raw_deltas)):.4f}  "
          f"sd={np.std(raw_deltas):.4f}")
    print(f"  DiD residual (deterioration) |D|: mean={np.mean(np.abs(did_resid)):.4f}  "
          f"sd={np.std(did_resid):.4f}")
    vr = 1 - np.var(did_resid)/np.var(raw_deltas)
    print(f"  variance absorbed by corridor: {100*vr:.1f}%  "
          f"(raw var {np.var(raw_deltas):.4f} -> DiD var {np.var(did_resid):.4f})")

    # parallel-trends probe: agreement of per-route raw deltas within a pair
    print("\n  parallel-trends probe (per-pair sd of route raw-deltas; small=common trend):")
    for (t0, t1) in pairs:
        ds = [C[(t1, r)] - C[(t0, r)] for r in ROUTES if (t0, r) in C and (t1, r) in C]
        print(f"    {CDATE[t0]}->{CDATE[t1]}: routes sd={np.std(ds):.4f} "
              f"corridor delta={corridorC[t1]-corridorC[t0]:+.4f}")

    # ===== (POS) semi-synthetic progressive defect recovery on 13th =====
    print("\n=== (POS) Semi-synthetic deterioration recovery (route 13th) ===")
    # pick a GPS location present in all 4 campaigns of 13th: use the medment
    # of overlapping cells. Inject growing roughness there across chronological
    # campaigns; DiD-correct with corridor; check monotone recovery vs controls.
    inj_factor = {"autumn": 1.0, "winter": 1.2, "spring": 1.4, "summer": 1.6}
    # choose injection location = median lat/lon of 13th winter ride mid-segment
    wx, wla, wlo = load_route("winter", "13th")
    nW = (len(wx)-X.WIN)//X.STEP + 1
    la, lo = win_gps(wla, wlo, nW)
    ok = ~(np.isnan(la) | np.isnan(lo))
    loc_lat, loc_lon = np.nanmedian(la[ok]), np.nanmedian(lo[ok])
    RAD = 0.00025   # ~25 m in degrees

    def near(la, lo):
        return (np.abs(la - loc_lat) < RAD) & (np.abs(lo - loc_lon) < RAD)

    prows = []
    for camp in CAMPS:
        xyz, laR, loR = load_route(camp, "13th")
        n = (len(xyz)-X.WIN)//X.STEP + 1
        wla, wlo = win_gps(laR, loR, n)
        inj_mask = near(wla, wlo)
        # inject: amplify raw accel in rows belonging to injected windows
        xyz_inj = xyz.copy()
        g = inj_factor[camp]
        rowmask = np.zeros(len(xyz), bool)
        for i in np.where(inj_mask)[0]:
            rowmask[i*X.STEP:i*X.STEP+X.WIN] = True
        xyz_inj[rowmask] = xyz_inj[rowmask].mean(0) + g*(xyz_inj[rowmask]-xyz_inj[rowmask].mean(0))
        Cclean = cond_index(xyz, fr)
        Cinj = cond_index(xyz_inj, fr)
        # localized condition at injected loc vs control (rest of route);
        # ALSO the injected location scored on the un-injected (clean) signal,
        # so we can isolate the pure injection effect from whatever natural
        # drift that specific GPS location already has (paired comparison).
        ctrl_mask = ~inj_mask & ~np.isnan(wla)
        prows.append(dict(camp=CDATE[camp], factor=g,
                          n_inj=int(inj_mask.sum()), n_ctrl=int(ctrl_mask.sum()),
                          C_injloc=float(np.nanmedian(Cinj[inj_mask])) if inj_mask.sum() else np.nan,
                          C_injloc_clean=float(np.nanmedian(Cclean[inj_mask])) if inj_mask.sum() else np.nan,
                          C_ctrl=float(np.nanmedian(Cinj[ctrl_mask])),
                          corridorC=corridorC[camp]))
    pdf = pd.DataFrame(prows)
    # DiD-correct against corridor, relative to first (autumn) campaign
    base = pdf.iloc[0]
    pdf["D_injloc"] = (pdf.C_injloc - base.C_injloc) - (pdf.corridorC - base.corridorC)
    pdf["D_injloc_clean"] = (pdf.C_injloc_clean - base.C_injloc_clean) - (pdf.corridorC - base.corridorC)
    pdf["D_ctrl"] = (pdf.C_ctrl - base.C_ctrl) - (pdf.corridorC - base.corridorC)
    pdf["D_injloc_minus_clean"] = pdf.D_injloc - pdf.D_injloc_clean   # paired: pure injection effect
    pdf.to_csv(X.OUT / "DET_synthetic_recovery.csv", index=False)
    print(pdf[["camp", "factor", "C_injloc", "C_injloc_clean", "C_ctrl", "corridorC",
               "D_injloc", "D_injloc_clean", "D_ctrl", "D_injloc_minus_clean"]].round(4).to_string(index=False))
    rho_inj = spearmanr(pdf.factor, pdf.D_injloc).correlation
    rho_paired = spearmanr(pdf.factor, pdf.D_injloc_minus_clean).correlation
    rho_ctrl = spearmanr(pdf.factor, pdf.D_ctrl).correlation
    print(f"\n  Spearman(injected factor, DiD@injected loc) = {rho_inj:+.3f}  (want ~+1)")
    print(f"  Spearman(injected factor, paired injected-minus-clean) = {rho_paired:+.3f}  (want ~+1)")
    print(f"  Spearman(injected factor, DiD@control loc)  = {rho_ctrl:+.3f}  (want ~0)")
    print(f"  saved -> DET_did_variance.csv, DET_synthetic_recovery.csv")


if __name__ == "__main__":
    main()
