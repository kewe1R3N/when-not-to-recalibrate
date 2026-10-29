#!/usr/bin/env python3
"""
Baseline comparison: W1-ACAS (Martinez Gil et al. 2026, arXiv:2604.20122,
Algorithm 1) applied to the exact same frozen-detector score streams and
calibrate/test split used by aic2_recal_table.py, so it slots in as an
additional row of Table~\\ref{tab:recal}.

Protocol (identical cal/test split to the other methods in aic2_recal_table.py):
  - For each shifted context, take the frozen-normalized smoothed score sm
    (same source-corridor normalization fr_src as all other methods).
  - Stream W1-ACAS over [cal_half, test_half] in chronological order (the
    method is online: it burns in and adapts its weights during cal_half,
    then continues adapting through test_half).
  - Flag window t as anomalous iff beta_t < ALPHA.
  - calib_err = |observed flagged-rate on test_half - ALPHA|, directly
    comparable to the calib_err column already reported for frozen/z-score/
    MAD/conformal/codebook+conformal.

Honesty notes carried into the paper text:
  - We apply Algorithm 1 (the general conformal-anomaly-score machinery,
    Sec. 3-5 of the source paper) directly to our own frozen detector's
    score stream; we do NOT reimplement the paper's Sec. 6 TSFM forecaster,
    since our score already plays the role of their nonconformity score S_t
    (see w1_acas.py docstring for the exact justification).
  - Hyperparameters (n, n_b, gamma) are ours, chosen to fit our much shorter
    per-context window counts (~300-1000) versus the source paper's presumably
    longer benchmark series; we report the exact values used and do not claim
    to reproduce their published numbers, only their algorithm.
"""
import numpy as np
import aic2_exp as X
from w1_acas import w1_acas

ALPHA = X.ALPHA  # 0.01, same budget as the rest of the paper


halves = X.halves          # guard-banded cal/test split (no smoothing-window bleed)


def main():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = X.A.load_model(X.CKPT, dev)

    SOURCE = "summer"
    CONTEXTS = [("season", "spring"), ("season", "autumn"), ("season", "winter"),
                ("device", "android")]

    def ride(model, path):
        data, _ = X.load_xyz(path)
        n = (len(data) - X.WIN) // X.STEP + 1
        E, D = X.raw_ED(model, data, dev)
        return dict(data=data, n=n, E=E, D=D)

    src = ride(m, str(X.IPHONE / SOURCE / "Nst.csv"))
    scal, _ = halves(src["n"])
    fr_src = X.calibrate(src["E"], src["D"], scal)

    print(f"W1-ACAS hyperparameters: alpha_c={ALPHA}, n_c={int(np.ceil(1/ALPHA-1))}")
    print(f"{'context':<10} {'n_windows':>10} {'n_cal':>7} {'n_test':>7} "
          f"{'flagged_rate':>13} {'calib_err':>10}")

    rows = []
    for axis, ctx in CONTEXTS:
        path = str((X.IPHONE / ctx / "Nst.csv") if axis == "season" else X.ANDROID_NST)
        r = ride(m, path)
        cal, tst = halves(r["n"])
        sm = X.apply_frozen(r["E"], r["D"], fr_src)

        n_windows = r["n"]
        n_cal = int(cal.sum())
        # size the online window/batch to the available calibration data
        n_past = max(50, min(400, n_cal))
        n_b = max(5, min(20, n_cal // 10))

        beta, diag = w1_acas(sm, alpha_c=ALPHA, n=n_past, n_b=n_b, gamma=0.05)
        test_beta = beta[tst]
        test_valid = ~np.isnan(test_beta)
        flagged_rate = float(np.mean(test_beta[test_valid] < ALPHA)) if test_valid.sum() else float("nan")
        calib_err = abs(flagged_rate - ALPHA)

        print(f"{ctx:<10} {n_windows:>10} {n_cal:>7} {int(tst.sum()):>7} "
              f"{flagged_rate:>13.4f} {calib_err:>10.4f}")
        rows.append(dict(axis=axis, context=ctx, method="W1-ACAS (Martinez Gil 2026)",
                          n_windows=n_windows, n_cal=n_cal, n_test=int(tst.sum()),
                          flagged_rate=round(flagged_rate, 4), calib_err=round(calib_err, 4),
                          n_past=n_past, n_b=n_b))

    import pandas as pd
    df = pd.DataFrame(rows)
    out = X.OUT / "B_w1acas_baseline.csv"
    df.to_csv(out, index=False)
    print(f"\nmean calibration error (W1-ACAS): {df['calib_err'].mean():.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
