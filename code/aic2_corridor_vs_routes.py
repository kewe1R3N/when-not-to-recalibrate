#!/usr/bin/env python3
"""Reference corridor vs other calibration sources (refine-logs/PREREG_corridor_vs_routes.md)."""
import numpy as np, pandas as pd, torch
import aic2_exp as X
from aic2_recal_table import ride

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.01
CTX = {"autumn": "Oct 2023", "winter": "Nov-Dec 2023", "spring": "Jan 2024", "android": "Android"}
ROUTES = ["11th", "13th", "16th", "billy"]
ANDROID_DIR = X.CODE.parent / "infer" / "device_comparison"


def path(ctx, route):
    if ctx == "android":
        p = ANDROID_DIR / f"android_{route}.csv"
    else:
        p = X.IPHONE / ctx / f"{route}.csv"
    return p if p.exists() else None


def main():
    m = X.A.load_model(X.CKPT, dev)
    src = ride(m, str(X.IPHONE / "summer" / "Nst.csv"))
    scal, _ = X.halves(src["n"])
    fr = X.calibrate(src["E"], src["D"], scal)
    rows, c3 = [], []
    for ctx, lab in CTX.items():
        nst = ride(m, str(X.ANDROID_NST if ctx == "android" else X.IPHONE / ctx / "Nst.csv"))
        cal, tst = X.halves(nst["n"])
        sn = X.apply_frozen(nst["E"], nst["D"], fr)
        ev = sn[tst]

        def rec(source, vals, extra=None):
            vals = vals[np.isfinite(vals)]
            tau = X.conformal_tau(vals, ALPHA)
            cer = float(np.mean(ev > tau)) if np.isfinite(tau) else 0.0
            rows.append(dict(context=lab, source=source, n_cal=len(vals), tau=round(float(tau), 4),
                             CER=round(cer, 4), dev_pp=round(100 * abs(cer - ALPHA), 2), **(extra or {})))
        rows.append(dict(context=lab, source="frozen", n_cal=None, tau=round(fr["tau"], 4),
                         CER=round(float(np.mean(ev > fr["tau"])), 4),
                         dev_pp=round(100 * abs(float(np.mean(ev > fr["tau"])) - ALPHA), 2)))
        rec("C0 corridor", sn[cal], dict(med_V4=round(float(np.nanmedian(nst["V"])), 4),
                                          med_score=round(float(np.nanmedian(sn)), 3)))
        pool = []
        n0 = int(cal.sum())
        for r in ROUTES:
            p = path(ctx, r)
            if p is None:
                continue
            rr = ride(m, str(p)); s = X.apply_frozen(rr["E"], rr["D"], fr)
            rec(f"C1 {r}", s, dict(med_V4=round(float(np.nanmedian(rr["V"])), 4),
                                    med_score=round(float(np.nanmedian(s)), 3)))
            pool.append(s)
            if len(s) >= n0:                                   # C3 size-matched contiguous blocks
                devs = []
                for st in range(0, len(s) - n0 + 1, 10):
                    tau = X.conformal_tau(s[st:st + n0][np.isfinite(s[st:st + n0])], ALPHA)
                    devs.append(100 * abs(float(np.mean(ev > tau)) - ALPHA))
                c3.append(dict(context=lab, route=r, n_blocks=len(devs), n_cal=n0,
                               dev_pp_mean=round(float(np.mean(devs)), 2),
                               dev_pp_q05=round(float(np.percentile(devs, 5)), 2),
                               dev_pp_q95=round(float(np.percentile(devs, 95)), 2)))
        rec("C2 pooled routes", np.concatenate(pool))
        print(f"[{lab}] done")
    df = pd.DataFrame(rows); df.to_csv(X.OUT / "CVR_sources.csv", index=False)
    d3 = pd.DataFrame(c3); d3.to_csv(X.OUT / "CVR_sizematched.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 80):
        print(df.to_string(index=False)); print(d3.to_string(index=False))
        piv = df.pivot_table(index="source", columns="context", values="dev_pp")
        piv["mean"] = piv.mean(axis=1); print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
