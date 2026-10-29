#!/usr/bin/env python3
"""
FI — systematic fault-injection study (Paper-2 third axis).

Motivation: v1 accidentally demonstrated that a silent preprocessing-convention
fault (missing standardization) causes catastrophic silent failure that a
corridor re-ride detects label-free. All three conventions exist in this
project's own wild data (Lincoln g-units, Charleston m/s^2 + gravity-on-Y,
Citadel gravity-removed). Here we inject faults DELIBERATELY and characterize:

  DETECT  : corridor FPR under frozen source thresholds (budget violation),
            corridor rho(s_t, VDV) (physical-proxy health check),
            VQ code-usage Jensen-Shannon divergence vs source (drift trigger, R009)
  TRIAGE  : Class A (per-axis affine: unit change, gravity/offset removal) should
            be NEUTRALIZED BY DESIGN by per-ride z-scoring at model entry
            (zscore(a*x+b) == zscore(x)); Class B (structural: missing-std stack
            fault, axis permutation, sampling-rate mismatch) is not.
  REMEDY  : conformal threshold recal on the faulted corridor (does FPR restore?)
            vs codebook re-anchoring (does route-level ranking restore?).

Protocol: source = clean summer/Nst corridor (first half calibrates fr_src +
code-usage reference). Each fault is applied to the summer corridor (detection +
threshold remedy) and to all summer routes (ranking + codebook remedy) —
pure-fault effect, no season confound. VDV is computed on the faulted stream
(deployment-real: the fault corrupts everything downstream of the sensor).
"""
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
import aic2_exp as X
import aic2_codebook as CB

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = X.ALPHA
ROUTES = ["11th", "13th", "16th", "Nst", "billy"]
CAMP = "summer"

# fault name -> (transform, standardize_flag, class)
FAULTS = {
    "none":         (lambda d: d,                 True,  "-"),
    "unit_ms2":     (lambda d: d * 9.81,          True,  "A (affine)"),
    "gravity_rm":   (lambda d: d - d.mean(0),     True,  "A (affine)"),
    "missing_std":  (lambda d: d,                 False, "B (stack)"),
    "axswap_YZ":    (lambda d: d[:, [0, 2, 1]],   True,  "B (orientation)"),
    "axswap_XY":    (lambda d: d[:, [1, 0, 2]],   True,  "B (orientation)"),
    "rate_2x":      (lambda d: d[::2],            True,  "B (temporal)"),
}


def code_indices(m, data, standardize=True):
    ze = CB.encode_ze(m, data, standardize=standardize)
    with torch.no_grad():
        z = torch.from_numpy(ze).float().to(dev)
        w = m.quant.embed.weight
        idx = []
        for i in range(0, len(z), 8192):
            idx.append(torch.cdist(z[i:i + 8192], w).argmin(1).cpu().numpy())
    return np.concatenate(idx)


def usage_hist(idx, K):
    h = np.bincount(idx, minlength=K).astype(float)
    return h / h.sum()


def js_div(p, q):
    def kl(a, b):
        mset = a > 0
        return float(np.sum(a[mset] * np.log(a[mset] / np.maximum(b[mset], 1e-12))))
    mid = (p + q) / 2
    return 0.5 * kl(p, mid) + 0.5 * kl(q, mid)


def st_rho(m, data, standardize=True):
    """(unsmoothed self-norm s_t, VDV, rho) on a (possibly faulted) ride."""
    n = CB.windows(data)
    E, D = X.raw_ED(m, data, dev, standardize=standardize)
    st = CB.st_selfnorm(E, D)
    V = CB.proxies(data, n)
    ok = ~(np.isnan(st) | np.isnan(V))
    return float(spearmanr(st[ok], V[ok]).correlation)


def main():
    m = X.A.load_model(X.CKPT, dev)
    orig_cb = m.quant.embed.weight.detach().clone()
    K = orig_cb.shape[0]

    # ---- clean source calibration + code-usage reference (summer corridor) ----
    cdata, _ = X.load_xyz(str(X.IPHONE / CAMP / "Nst.csv"))
    n0 = CB.windows(cdata)
    E0, D0 = X.raw_ED(m, cdata, dev)
    cal0, _ = X.halves(n0)
    fr_src = X.calibrate(E0, D0, cal0)
    p_src = usage_hist(code_indices(m, X.cal_raw_slice(cdata, int(cal0.sum()))), K)
    print(f"source = clean {CAMP}/Nst corridor: n={n0}, tau={fr_src['tau']:.4f}")

    route_data = {}
    for r in ROUTES:
        f = X.IPHONE / CAMP / f"{r}.csv"
        if f.exists():
            route_data[r] = X.load_xyz(str(f))[0]

    rows = []
    for name, (tf, std_flag, cls) in FAULTS.items():
        cf = tf(cdata.copy())
        nf = CB.windows(cf)
        cal, tst = X.halves(nf)

        # DETECT signals on the corridor
        E, D = X.raw_ED(m, cf, dev, standardize=std_flag)
        sm = X.apply_frozen(E, D, fr_src)
        fpr_frozen = float(np.mean(sm[tst] > fr_src["tau"]))
        rho_corr = st_rho(m, cf, standardize=std_flag)
        # Slice the same guard-banded window-count prefix as p_src (not a naive
        # raw-sample half): with a different convention, even the unfaulted
        # "none" condition compared against itself picked up a tiny spurious
        # JS (~0.0001) from the two prefixes covering slightly different
        # numbers of windows, which is not a real signal.
        p_f = usage_hist(code_indices(m, X.cal_raw_slice(cf, int(cal.sum())), standardize=std_flag), K)
        js = js_div(p_f, p_src)

        # REMEDY 1: conformal threshold recal on faulted corridor
        tau_c = X.conformal_tau(sm[cal], ALPHA)
        fpr_conf = float(np.mean(sm[tst] > tau_c))

        # route-level ranking, frozen codebook
        rhos_frozen = [st_rho(m, tf(route_data[r].copy()), standardize=std_flag)
                       for r in route_data]

        # REMEDY 2: codebook re-anchor on faulted corridor -> faulted routes.
        # Fit only on the calibration-half raw samples (X.cal_raw_slice) — the
        # full-ride fit used here previously leaked the test half into the
        # k-means fit that FPR/rho are then evaluated against.
        # Standardize (if std_flag) the whole faulted corridor first, matching
        # how the rest of this script scores `cf`, then slice the calibration
        # prefix from that already-transformed array so the codebook fit sees
        # the same per-sample transform as corridor-level scoring does.
        cf_for_fit = X.zscore(cf) if std_flag else np.asarray(cf, np.float32)
        cal_raw_f = X.cal_raw_slice(cf_for_fit, int(cal.sum()))
        ze = CB.encode_ze(m, cal_raw_f, standardize=False)
        m.quant.embed.weight.data.copy_(CB.refit_codebook(m, ze))
        rhos_cb = [st_rho(m, tf(route_data[r].copy()), standardize=std_flag)
                   for r in route_data]
        m.quant.embed.weight.data.copy_(orig_cb)

        rows.append(dict(fault=name, cls=cls,
                         FPR_frozen=round(fpr_frozen, 4),
                         FPR_x=round(fpr_frozen / ALPHA, 1),
                         rho_corridor=round(rho_corr, 3),
                         JS_usage=round(js, 4),
                         FPR_conformal=round(fpr_conf, 4),
                         rho_routes_frozen=round(float(np.mean(rhos_frozen)), 3),
                         rho_routes_cbfix=round(float(np.mean(rhos_cb)), 3)))
        print(f"[{name:12s}] {cls:16s} FPR {fpr_frozen:.3f} ({fpr_frozen/ALPHA:.0f}x)  "
              f"rho_corr {rho_corr:+.3f}  JS {js:.4f}  | conf-FPR {fpr_conf:.3f}  "
              f"routes rho {np.mean(rhos_frozen):+.3f} -> cb {np.mean(rhos_cb):+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(X.OUT / "FI_fault_grid.csv", index=False)
    print("\n=== fault-injection grid ===")
    print(df.to_string(index=False))
    print(f"\nsaved -> {X.OUT/'FI_fault_grid.csv'}")


if __name__ == "__main__":
    main()
