#!/usr/bin/env python3
"""Detection retention under recalibrated thresholds, Jan 2024 (refine-logs/PREREG_detection_retention.md)."""
import numpy as np, pandas as pd, torch
import aic2_exp as X
import aic2_codebook as CB
from aic2_recal_table import ride

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LORO = X.CODE / "latex_output" / "loro_results"
FOLDS = {"13th": "fold_1_13th", "billy": "fold_3_Billy", "Nst": "fold_4_N"}


def main():
    m = X.A.load_model(X.CKPT, dev)
    src = ride(m, str(X.IPHONE / "summer" / "Nst.csv"))
    scal, _ = X.halves(src["n"]); fr = X.calibrate(src["E"], src["D"], scal)
    cvr = pd.read_csv(X.OUT / "CVR_sources.csv")
    jan = cvr[cvr.context == "Jan 2024"].set_index("source")
    T = {"frozen": fr["tau"], "corridor": float(jan.loc["C0 corridor", "tau"]),
         "pooled routes": float(jan.loc["C2 pooled routes", "tau"])}
    # recompute corridor/pooled taus at full precision rather than the rounded CSV values
    nst = ride(m, str(X.IPHONE / "spring" / "Nst.csv")); cal, tst = X.halves(nst["n"])
    sn = X.apply_frozen(nst["E"], nst["D"], fr); T["corridor"] = X.conformal_tau(sn[cal], X.ALPHA)
    pool = []
    for r in ["11th", "13th", "16th", "billy"]:
        rr = ride(m, str(X.IPHONE / "spring" / f"{r}.csv")); s = X.apply_frozen(rr["E"], rr["D"], fr)
        pool.append(s[np.isfinite(s)])
    T["pooled routes"] = X.conformal_tau(np.concatenate(pool), X.ALPHA)
    jan_cer = float(np.mean(sn[tst] > fr["tau"]))
    S, Y = {}, {}
    for r, f in FOLDS.items():
        data, anom = X.load_xyz(str(LORO / f / "data" / "test.csv"))
        n = CB.windows(data); y = ~X.window_normal_mask(anom, n)
        E, D = X.raw_ED(m, data, dev); s = X.apply_frozen(E, D, fr)
        ok = np.isfinite(s); S[r], Y[r] = s[ok], y[ok]
    rows = []
    groups = {"13th+Billy (primary)": ["13th", "billy"], "13th": ["13th"], "Billy": ["billy"], "N St (secondary, overlap)": ["Nst"]}
    for g, rs in groups.items():
        s = np.concatenate([S[r] for r in rs]); y = np.concatenate([Y[r] for r in rs])
        base = None
        for name, tau in T.items():
            a = s > tau; tp = int((a & y).sum())
            rec = tp / y.sum(); prec = tp / a.sum() if a.sum() else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if a.sum() and (prec + rec) else 0.0
            if name == "frozen": base = rec
            rows.append(dict(group=g, threshold=name, tau=round(float(tau), 4), n_windows=len(y), n_pos=int(y.sum()),
                             alarm_rate=round(float(a.mean()), 4), recall=round(rec, 4),
                             precision=round(prec, 4) if a.sum() else None, F1=round(f1, 4),
                             recall_retention=round(rec / base, 3) if base else None,
                             neg_alarm_rate=round(float(a[~y].mean()), 4)))
    df = pd.DataFrame(rows); df.to_csv(X.OUT / "DET_retention.csv", index=False)
    with pd.option_context("display.width", 220):
        print(df.to_string(index=False))
    print(f"sanity: Jan corridor held-out CER under frozen = {jan_cer:.4f}")


if __name__ == "__main__":
    main()
