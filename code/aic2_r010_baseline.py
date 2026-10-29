#!/usr/bin/env python3
"""
R010 + physical baseline — the two remaining reviewer defenses.

(A) R010 CORRIDOR-LENGTH CURVE: how many corridor calibration windows does the
    conformal P99 threshold need to control a 1% budget?  For each context,
    subsample CONTIGUOUS blocks of n_cal windows from the calibration half
    (temporal dependence respected), set tau = empirical P99, evaluate FPR on
    the fixed held-out half; B random block positions -> FPR distribution vs
    n_cal.  Explains the Citadel revisit failure at n_cal=85 and yields the
    protocol's minimum-corridor design rule.  iid-theory reference: with n cal
    points, split-conformal expected FPR = 1 - ceil((1-a)(n+1))/(n+1).

(B) PHYSICAL BASELINE ("do you need the learned model at all?"):
    (B1) calibration lane: raw VDV as the score, same split-half conformal
         protocol, same contexts as B23 -> calibration error comparison.
    (B2) detection lane (labeled loro folds 1/3/4): PR-AUC of physical scores
         (VDV, crest factor, RMS) vs the frozen learned s_t on ground-truth
         window labels.  This is the lane where the model must earn its keep.
"""
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
import aic2_exp as X
import aic2_codebook as CB

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = X.ALPHA
RNG = np.random.RandomState(42)
CONTEXTS = ["summer", "spring", "autumn", "winter", "android"]
NCAL_GRID = [25, 50, 85, 150, 250, 350, 500]
B = 200
LORO = X.CODE / "latex_output" / "loro_results"
FOLDS = {"13th": "fold_1_13th", "billy": "fold_3_Billy", "Nst": "fold_4_N"}


def load_ctx(m, ctx):
    path = X.ANDROID_NST if ctx == "android" else str(X.IPHONE / ctx / "Nst.csv")
    data, _ = X.load_xyz(path)
    n = CB.windows(data)
    E, D = X.raw_ED(m, data, dev)
    V = CB.proxies(data, n)
    return dict(data=data, n=n, E=E, D=D, V=V)


def main():
    m = X.A.load_model(X.CKPT, dev)
    cache = {c: load_ctx(m, c) for c in CONTEXTS}

    # Lincoln summer normalization (B23 convention)
    src = cache["summer"]
    scal = np.zeros(src["n"], bool); scal[: src["n"] // 2] = True
    fr_lin = X.calibrate(src["E"], src["D"], scal)

    # ---------------- (A) R010 corridor-length curve ----------------
    print("=== (A) R010: FPR vs calibration-corridor size (B=200 contiguous blocks) ===")
    rows = []
    guard = X.RAW_INDEP_GUARD
    for ctx in CONTEXTS:
        r = cache[ctx]
        H = r["n"] // 2
        Hc = H - guard          # cal-sampling region ends here (guard-banded)
        sm = X.apply_frozen(r["E"], r["D"], fr_lin)
        tst = np.zeros(r["n"], bool); tst[min(r["n"], H + guard):] = True
        for ncal in [g for g in NCAL_GRID if g <= Hc]:
            fprs = np.empty(B)
            for b in range(B):
                s = RNG.randint(0, Hc - ncal + 1)
                tau = X.conformal_tau(sm[s:s + ncal], ALPHA)
                fprs[b] = float(np.mean(sm[tst] > tau))
            exp_iid = 1 - np.ceil((1 - ALPHA) * (ncal + 1)) / (ncal + 1)
            rows.append(dict(context=ctx, n_cal=ncal,
                             FPR_mean=round(float(fprs.mean()), 4),
                             FPR_q05=round(float(np.quantile(fprs, .05)), 4),
                             FPR_q95=round(float(np.quantile(fprs, .95)), 4),
                             P_FPR_gt_3pct=round(float(np.mean(fprs > .03)), 3),
                             expected_iid=round(float(exp_iid), 4)))
        got = [x for x in rows if x["context"] == ctx]
        print(f"  [{ctx:7s}] " + "  ".join(
            f"n={g['n_cal']}: {g['FPR_mean']:.3f}[{g['FPR_q05']:.3f},{g['FPR_q95']:.3f}]"
            for g in got))
    df = pd.DataFrame(rows)
    df.to_csv(X.OUT / "R010_corridor_length.csv", index=False)

    # ---------------- (B1) physical-score calibration lane ----------------
    print("\n=== (B1) VDV-threshold baseline: calibration error (same protocol as B23) ===")
    b1 = []
    for ctx in ["spring", "autumn", "winter", "android"]:
        r = cache[ctx]
        cal, tst = X.halves(r["n"])       # guard-banded: no smoothing-window bleed
        Vs = X.smooth(r["V"])
        tauV = X.conformal_tau(Vs[cal], ALPHA)
        fpr = float(np.nanmean(Vs[tst] > tauV))
        b1.append(dict(context=ctx, method="VDV+conformal",
                       FPR=round(fpr, 4), calib_err=round(abs(fpr - ALPHA), 4)))
        print(f"  {ctx:8s} VDV-conformal FPR {fpr:.4f}  |err| {abs(fpr-ALPHA):.4f}")
    dfb1 = pd.DataFrame(b1)
    dfb1.to_csv(X.OUT / "BASE_physical_calibration.csv", index=False)
    print(f"  mean calib err (VDV) = {dfb1.calib_err.mean():.4f}   "
          f"(learned+conformal was 0.0202, frozen 0.0396)")

    # ---------------- (B2) physical vs learned: labeled detection lane ----------------
    print("\n=== (B2) labeled loro folds: PR-AUC physical scores vs learned s_t ===")
    b2 = []
    for name, fold in FOLDS.items():
        data, anom = X.load_xyz(str(LORO / fold / "data" / "test.csv"))
        n = CB.windows(data)
        y = (~X.window_normal_mask(anom, n)).astype(int)
        E, D = X.raw_ED(m, data, dev)
        scores = {
            "learned s_t": X.smooth(CB.st_selfnorm(E, D)),
            "VDV": X.smooth(CB.proxies(data, n)),
            "crest (SHI)": X.smooth(X.crest_per_window(data, n)),
            "RMS |a|": X.smooth(np.array([
                float(np.sqrt(np.mean(np.linalg.norm(
                    data[i*X.STEP:i*X.STEP+X.WIN], axis=1) ** 2)))
                for i in range(n)])),
        }
        for meth, s in scores.items():
            ok = ~np.isnan(s)
            b2.append(dict(fold=name, method=meth,
                           PR_AUC=round(average_precision_score(y[ok], s[ok]), 4),
                           ROC_AUC=round(roc_auc_score(y[ok], s[ok]), 4)))
        print(f"  [{name}] " + "  ".join(
            f"{r['method']}: AP={r['PR_AUC']:.3f}" for r in b2[-4:]))
    dfb2 = pd.DataFrame(b2)
    dfb2.to_csv(X.OUT / "BASE_prauc_physical.csv", index=False)
    print("\n  PR-AUC pivot:")
    print(dfb2.pivot_table(index="method", columns="fold", values="PR_AUC").round(3).to_string())
    print("  mean PR-AUC per method:")
    print(dfb2.groupby("method")["PR_AUC"].mean().round(3).to_string())
    print(f"\nsaved -> R010_corridor_length.csv, BASE_physical_calibration.csv, "
          f"BASE_prauc_physical.csv")


if __name__ == "__main__":
    main()
