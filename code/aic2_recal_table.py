#!/usr/bin/env python3
"""
Block B2/B3 — centerpiece table: recalibration methods x shifted contexts.

All methods operate on the FROZEN VQTransAE win65. They differ only in how they
set the operating point (and, for the codebook method, in re-anchoring the VQ
codebook). No labels are used: calibration error is measured on corridor-normal
windows; ranking uses the VDV physical proxy (Paper-1 validation style). Real GT
labels exist only for the in-domain loro iPhone routes (no labelled season/device
data), so PR-AUC-under-shift is intentionally NOT claimed — calibration error is
the primary, fully label-free metric the paper is about.

Methods:
  frozen              : source (summer) normalization + source tau           (no adaptation)
  z-score             : threshold-only, parametric  tau = mean + 2.326 sd    (target corridor)
  MAD                 : threshold-only, robust       tau = med  + 3.45 MAD    (target corridor)
  conformal (ours)    : threshold-only, split-conformal empirical P99         (target corridor)
  codebook+conformal  : re-estimate VQ codebook on target corridor + conformal tau  (ours-full)

Metric (target alpha = 1%): |FPR - alpha| on the target corridor TEST half
(split-half: calibrate on first half, evaluate on second half). Ranking rho(s_t,VDV)
is identical for all threshold-only methods (monotone) and only moves for codebook.
"""
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, norm
import aic2_exp as X
import aic2_codebook as CB

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.01
Z99 = norm.ppf(1 - ALPHA)                 # 2.326
MAD_K = Z99 / 0.6745                       # ~3.449 (normal-consistent)
SOURCE = "summer"
CONTEXTS = [("season", "spring"), ("season", "autumn"), ("season", "winter"),
            ("device", "android")]


def ride(model, path):
    data, _ = X.load_xyz(path)
    n = (len(data) - X.WIN) // X.STEP + 1
    E, D = X.raw_ED(model, data, dev)
    V = np.full(n, np.nan)
    for i in range(n):
        s = i * X.STEP; w = data[s:s + X.WIN]
        V[i] = X.A.compute_window_proxies(w[:, 0], w[:, 1], w[:, 2], X.FS,
                                          X.A.config.HP_CUTOFF)["VDV_w"]
    return dict(data=data, n=n, E=E, D=D, V=V)


def svals(E, D, fr):
    return X.apply_frozen(E, D, fr)        # smoothed s_t under normalization fr


halves = X.halves          # guard-banded cal/test split (no smoothing-window bleed)


def fpr(sm, tau, mask):
    return float(np.mean(sm[mask] > tau))


def rho(sm, V):
    ok = ~(np.isnan(sm) | np.isnan(V))
    return spearmanr(sm[ok], V[ok]).correlation


def main():
    m = X.A.load_model(X.CKPT, dev)
    orig_cb = m.quant.embed.weight.detach().clone()

    # --- source frozen calibration (summer corridor, first half) ---
    src = ride(m, str(X.IPHONE / SOURCE / "Nst.csv"))
    scal, _ = halves(src["n"])
    fr_src = X.calibrate(src["E"], src["D"], scal)          # E0/E1/D0/D1 + tau (P99)

    rows = []
    for axis, ctx in CONTEXTS:
        path = str((X.IPHONE / ctx / "Nst.csv") if axis == "season" else X.ANDROID_NST)
        r = ride(m, path)
        cal, tst = halves(r["n"])
        # frozen-normalized score (summer constants) — shared by all threshold methods
        sm = svals(r["E"], r["D"], fr_src)
        rho_thr = rho(sm, r["V"])                            # ranking (threshold-only)
        cal_vals = sm[cal]

        methods = {
            "frozen":    fr_src["tau"],
            "z-score":   float(np.nanmean(cal_vals) + Z99 * np.nanstd(cal_vals)),
            "MAD":       float(np.nanmedian(cal_vals) +
                               MAD_K * np.nanmedian(np.abs(cal_vals - np.nanmedian(cal_vals)))),
            "conformal": X.conformal_tau(cal_vals, ALPHA),
        }
        for name, tau in methods.items():
            p = fpr(sm, tau, tst)
            rows.append(dict(axis=axis, context=ctx, method=name, tau=round(tau, 3),
                             FPR=round(p, 4), calib_err=round(abs(p - ALPHA), 4),
                             rho_st_VDV=round(rho_thr, 3), params_updated=0,
                             needs_forward=False))

        # --- codebook + conformal (ours-full): re-estimate codebook on target corridor ---
        # Fit the codebook ONLY on the calibration-half raw samples (X.cal_raw_slice) —
        # fitting on the full ride would leak test-half windows into the k-means fit
        # that is then evaluated on those very same test-half windows.
        #
        # Normalization is held at fr_src (the SAME source-fixed E0/E1/D0/D1 every
        # other method in this table uses) rather than re-derived from the target
        # corridor. Re-deriving target-specific normalization here would confound
        # two simultaneous changes (codebook AND normalization scheme) in one row,
        # making it impossible to attribute any calibration-error change to the
        # codebook re-anchoring specifically rather than the normalization switch.
        # Standardize the WHOLE ride first (matching exactly what raw_ED() does
        # for scoring), then slice the calibration-half prefix from that ALREADY
        # z-scored array and tell encode_ze not to re-standardize. Re-standardizing
        # just the calibration prefix (as an earlier version did) computes a
        # different mean/std than scoring uses, so the codebook would be fit on
        # a different transform of the data than it is later scored on.
        m.quant.embed.weight.data.copy_(orig_cb)
        full_z = X.zscore(r["data"])
        cal_raw_z = X.cal_raw_slice(full_z, int(cal.sum()))
        ze = CB.encode_ze(m, cal_raw_z, standardize=False)
        m.quant.embed.weight.data.copy_(CB.refit_codebook(m, ze))
        E2, D2 = X.raw_ED(m, r["data"], dev)
        sm2 = svals(E2, D2, fr_src)                         # same normalization as all other rows
        tau2 = X.conformal_tau(sm2[cal], ALPHA)
        rows.append(dict(axis=axis, context=ctx, method="codebook+conformal",
                         tau=round(tau2, 3), FPR=round(fpr(sm2, tau2, tst), 4),
                         calib_err=round(abs(fpr(sm2, tau2, tst) - ALPHA), 4),
                         rho_st_VDV=round(rho(sm2, r["V"]), 3),
                         params_updated=int(orig_cb.numel()), needs_forward=True))
        m.quant.embed.weight.data.copy_(orig_cb)
        print(f"[{axis}/{ctx}] done")

    df = pd.DataFrame(rows)
    df.to_csv(X.OUT / "B23_recal_table.csv", index=False)

    print("\n=== calibration error |FPR-1%| (lower=better) ; rows=method, cols=context ===")
    piv = df.pivot_table(index="method", columns="context", values="calib_err")
    order = ["frozen", "z-score", "MAD", "conformal", "codebook+conformal"]
    cols = ["spring", "autumn", "winter", "android"]
    print(piv.reindex(index=order)[cols].to_string())
    print("\nmean calibration error per method:")
    print(df.groupby("method")["calib_err"].mean().reindex(order).round(4).to_string())
    print("\n=== ranking rho(s_t,VDV): threshold-only vs codebook ===")
    rr = df.pivot_table(index="method", columns="context", values="rho_st_VDV")
    print(rr.reindex(index=order)[cols].to_string())
    print(f"\nsaved -> {X.OUT/'B23_recal_table.csv'}")


if __name__ == "__main__":
    main()
