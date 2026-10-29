#!/usr/bin/env python3
"""
V2 battery — full re-run of the Paper-2 core under the CORRECTED input
convention: per-ride per-axis z-scoring at model entry, matching Paper-1's own
inference pipeline (inference_new_data.py L229-233).

Why: v1 (archived in refine-logs/aic2_results_v1_rawg_fault/) fed raw g-unit
campaign data into a model trained on standardized inputs. That single
convention fault produced the "0-52x FPR drift" and the "spring representation
collapse" — verified by timestamp-aligning loro fold-4 (standardized) with
spring/Nst (raw): corr=+1.0000 per axis, per-axis affine; and by the decisive
test (spring/Nst rho -0.191 raw -> +0.924 z-scored, all seasons 0.86-0.96).
v1 numbers are hereafter a pipeline-fault STRESS TEST, not season/device drift.

Stages:
  V2-M0   loro reproduction under the per-ride convention (tau vs Paper-1 0.659)
  V2-M1   leave-one-source-out corridor-FPR drift matrix (season x device)
  V2-RANK frozen-model rho(s_t, VDV_raw) on all campaign x route cells + android
  V2-CI   block-bootstrap 95% CIs, frozen vs conformal, per shifted context
  V2-B23  centerpiece methods x contexts table (calls aic2_recal_table.main())
"""
import numpy as np, pandas as pd, torch, json
from scipy.stats import spearmanr
import aic2_exp as X
import aic2_codebook as CB
import aic2_recal_table as RT
from aic2_finalize import block_bootstrap_ci

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMPS = ["spring", "summer", "autumn", "winter"]
ROUTES = ["11th", "13th", "16th", "Nst", "billy"]
ALPHA = X.ALPHA


def corridor_scores(m, camp):
    path = X.ANDROID_NST if camp == "android" else str(X.IPHONE / camp / "Nst.csv")
    data, _ = X.load_xyz(path)
    n = (len(data) - X.WIN) // X.STEP + 1
    E, D = X.raw_ED(m, data, dev)          # z-scored at model entry
    return dict(data=data, n=n, E=E, D=D)


def main():
    m = X.A.load_model(X.CKPT, dev)

    # ---------------- V2-M0: loro reproduction under per-ride convention ----
    print("\n########## V2-M0: loro N St reproduction (per-ride z-score) ##########")
    X.run_M0(m, dev)   # writes M0_reproduction.json to the fresh OUT dir

    # ---------------- V2-M1: drift matrix ----------------
    print("\n########## V2-M1: leave-one-source-out corridor FPR (target 1%) ##########")
    cache = {c: corridor_scores(m, c) for c in CAMPS + ["android"]}
    rows = []
    for src in CAMPS:
        r = cache[src]
        cal, tst = X.halves(r["n"])       # guard-banded: no smoothing-window bleed
        fr = X.calibrate(r["E"], r["D"], cal)
        sm_src = X.apply_frozen(r["E"], r["D"], fr)
        rows.append(dict(source=src, eval=f"{src}(held-out half)",
                         FPR=float(np.mean(sm_src[tst] > fr["tau"]))))
        for tgt in CAMPS + ["android"]:
            if tgt == src:
                continue
            t = cache[tgt]
            sm = X.apply_frozen(t["E"], t["D"], fr)
            rows.append(dict(source=src, eval=tgt, FPR=float(np.mean(sm > fr["tau"]))))
    df1 = pd.DataFrame(rows)
    df1["inflation_x"] = (df1["FPR"] / ALPHA).round(1)
    df1.to_csv(X.OUT / "M1v2_drift_matrix.csv", index=False)
    piv = df1[~df1["eval"].str.contains("held-out")].pivot_table(
        index="source", columns="eval", values="inflation_x")
    print(piv.reindex(index=CAMPS)[[c for c in CAMPS + ["android"] if c in piv.columns]]
          .to_string())
    print("(cells = FPR as multiples of the 1% target; v1 was 0x-52x)")

    # ---------------- V2-RANK: frozen ranking, all cells ----------------
    print("\n########## V2-RANK: frozen rho(s_t, VDV_raw), all campaigns x routes ##########")
    rows = []
    for camp in CAMPS:
        for rt in ROUTES:
            f = X.IPHONE / camp / f"{rt}.csv"
            if not f.exists():
                continue
            d, _ = X.load_xyz(str(f))
            n = CB.windows(d)
            if n < 30:
                continue
            E, D = X.raw_ED(m, d, dev)
            st = CB.st_selfnorm(E, D)
            V = CB.proxies(d, n)               # VDV from RAW physical data
            ok = ~(np.isnan(st) | np.isnan(V))
            rows.append(dict(camp=camp, route=rt, n=int(ok.sum()),
                             rho=round(float(spearmanr(st[ok], V[ok]).correlation), 3)))
    # android corridor + other android routes (files are already standardized;
    # z-score is idempotent; VDV computed on the standardized values, noted caveat).
    # android_Nst is cached as "android" already; the other 3 routes have their
    # own device_comparison/android_<route>.csv files, previously left out of
    # this grid even though they exist (an omission, not a deliberate scope cut).
    ANDROID_DIR = X.CODE.parent / "infer" / "device_comparison"
    for rt in ["11th", "16th", "billy"]:
        f = ANDROID_DIR / f"android_{rt}.csv"
        if not f.exists():
            continue
        d, _ = X.load_xyz(str(f))
        n = CB.windows(d)
        if n < 30:
            continue
        E, D = X.raw_ED(m, d, dev)
        st_r = CB.st_selfnorm(E, D)
        V_r = CB.proxies(d, n)
        ok_r = ~(np.isnan(st_r) | np.isnan(V_r))
        rows.append(dict(camp="android", route=rt, n=int(ok_r.sum()),
                         rho=round(float(spearmanr(st_r[ok_r], V_r[ok_r]).correlation), 3)))
    a = cache["android"]
    st = CB.st_selfnorm(a["E"], a["D"])
    V = CB.proxies(a["data"], a["n"])
    ok = ~(np.isnan(st) | np.isnan(V))
    rows.append(dict(camp="android", route="Nst", n=int(ok.sum()),
                     rho=round(float(spearmanr(st[ok], V[ok]).correlation), 3)))
    df2 = pd.DataFrame(rows)
    df2.to_csv(X.OUT / "M1cv2_ranking_allroutes.csv", index=False)
    print(df2.pivot_table(index="camp", columns="route", values="rho").round(3).to_string())
    print("(v1 raw-g spring row was 0.02-0.61 with Nst=-0.19)")

    # ---------------- V2-CI: block-bootstrap CIs ----------------
    print("\n########## V2-CI: frozen vs conformal corridor FPR, 95% block-bootstrap ##########")
    src = cache["summer"]
    scal, _ = X.halves(src["n"])
    fr_src = X.calibrate(src["E"], src["D"], scal)
    ci_rows = []
    for ctx in ["spring", "autumn", "winter", "android"]:
        r = cache[ctx]
        cal, tst = X.halves(r["n"])       # guard-banded: no smoothing-window bleed
        sm = X.apply_frozen(r["E"], r["D"], fr_src)
        pf, lof, hif = block_bootstrap_ci((sm[tst] > fr_src["tau"]).astype(float))
        tau_c = X.conformal_tau(sm[cal], ALPHA)
        pc, loc, hic = block_bootstrap_ci((sm[tst] > tau_c).astype(float))
        ci_rows.append(dict(context=ctx, frozen_FPR=round(pf, 4),
                            frozen_CI=f"[{lof:.4f},{hif:.4f}]",
                            conformal_FPR=round(pc, 4),
                            conformal_CI=f"[{loc:.4f},{hic:.4f}]"))
        print(f"  {ctx:8s} frozen {pf:.4f} CI[{lof:.4f},{hif:.4f}]   "
              f"conformal {pc:.4f} CI[{loc:.4f},{hic:.4f}]")
    pd.DataFrame(ci_rows).to_csv(X.OUT / "B_bootstrap_ci_v2.csv", index=False)

    # ---------------- V2-B23: centerpiece table ----------------
    print("\n########## V2-B23: recalibration methods x contexts ##########")
    RT.main()


if __name__ == "__main__":
    main()
