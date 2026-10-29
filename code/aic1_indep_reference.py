#!/usr/bin/env python3
"""
Paper-1 Option-1: adopt N 21st St (independent, NOT in the 4-route evaluation set)
as the smooth reference corridor and re-derive the frozen thresholds on it, then
apply them to the four evaluation routes.

Guardrail: N 21st St is a Sensor Logger ride (m/s^2, gravity/orientation streams),
a different convention from the study's X/Y/Z (g) data. Before trusting any
threshold we run a CONVENTION HEALTH CHECK: rho(s_t, VDV) on the reference must be
positive and comparable to the other routes; otherwise it is a convention fault
(as with Charleston Ravenel v1) and the thresholds are meaningless.

No manuscript edits — numbers only.
"""
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
import aic2_exp as X
from aic2_city import rot_to_minus_z

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MP = X.CODE.parent                      # master_project
REF = str(MP / "N_21st_St-29-05-2025_8-30-27.csv")
SEG = (550, 2615)                        # smoothest 401-window steady segment (t=55-262s)
ALPHA = 0.01


def load_sensorlogger_g(path, seg=None):
    """Sensor Logger accel -> reoriented, g-unit [N,3] matching the study convention.
    Per-sample rotation via the gravity stream so gravity -> -Z, then /9.81."""
    r = pd.read_csv(path)
    a = r[["accelerometer_x", "accelerometer_y", "accelerometer_z"]].values.astype(np.float64)
    g = r[["gravity_x", "gravity_y", "gravity_z"]].values.astype(np.float64)
    if seg:
        a, g = a[seg[0]:seg[1]], g[seg[0]:seg[1]]
    out = np.empty_like(a)
    for i in range(len(a)):
        out[i] = rot_to_minus_z(g[i]) @ a[i]
    return (out / 9.81).astype(np.float32)


def proxies_vdv(data, n):
    return np.array([X.A.compute_window_proxies(data[i*X.STEP:i*X.STEP+X.WIN, 0],
                                                data[i*X.STEP:i*X.STEP+X.WIN, 1],
                                                data[i*X.STEP:i*X.STEP+X.WIN, 2],
                                                X.FS, X.A.config.HP_CUTOFF)["VDV_w"]
                     for i in range(n)])


def main():
    m = X.A.load_model(X.CKPT, dev)

    # ---- build N 21st St reference input ----
    ref = load_sensorlogger_g(REF, SEG)
    n = (len(ref) - X.WIN)//X.STEP + 1
    E, D = X.raw_ED(m, ref, dev)                 # z-scored at entry
    sm = X.apply_frozen(E, D, X.calibrate(E, D, np.ones(n, bool)))  # self-normalized smoothed s_t
    V = proxies_vdv(ref, n)
    cf = X.crest_per_window(ref, n)
    ok = ~(np.isnan(sm) | np.isnan(V))
    rho = spearmanr(sm[ok], V[ok]).correlation

    print("=== CONVENTION HEALTH CHECK (N 21st St reference) ===")
    print(f"  windows={n}   rho(s_t, VDV)={rho:+.3f}   (healthy if >~0.5, cf. study routes 0.55-0.92)")
    if not (rho > 0.4):
        print("  !! UNHEALTHY -> convention fault; thresholds NOT trustworthy. STOP.")
        return
    print("  -> healthy; convention alignment OK, proceeding to threshold derivation.")

    # ---- frozen thresholds on the independent reference (Paper-1 definitions) ----
    fr = X.calibrate(E, D, np.ones(n, bool), cf)   # sets E0/E1/D0/D1, tau(P99), tau_cdi(P90), tau_shi(P90)
    tau_det = fr["tau"]; tau_cdi = fr["tau_cdi"]; tau_shi = fr["tau_shi"]
    print("\n=== FROZEN THRESHOLDS re-derived on N 21st St (independent reference) ===")
    print(f"  detection tau (P99 smoothed s_t) = {tau_det:.4f}   [Paper-1 on N St: 0.659]")
    print(f"  tau_CDI (P90 smoothed s_t)       = {tau_cdi:.4f}   [Paper-1 on N St: 0.43]")
    print(f"  tau_SHI (P90 crest factor)       = {tau_shi:.4f}   [Paper-1 on N St: 4.79]")

    # ---- apply frozen detection tau to the four evaluation routes ----
    # normalization constants (E0/E1/D0/D1) come from the reference; tau frozen.
    print("\n=== apply frozen (N-21st-St) thresholds to the 4 evaluation routes ===")
    folds = {"13th": "fold_1_13th", "16th": "fold_2_16th",
             "Billy": "fold_3_Billy", "N St": "fold_4_N"}
    from sklearn.metrics import f1_score, precision_score, recall_score
    rows = []
    for name, fold in folds.items():
        p = str(X.CODE / "latex_output" / "loro_results" / fold / "data" / "test.csv")
        try:
            data, anom = X.load_xyz(p)
        except Exception as e:
            print(f"  {name}: load error {e}"); continue
        nn = (len(data) - X.WIN)//X.STEP + 1
        Er, Dr = X.raw_ED(m, data, dev)
        smr = X.apply_frozen(Er, Dr, fr)          # reference normalization + frozen tau
        y = (~X.window_normal_mask(anom, nn)).astype(int)
        pred = (smr > tau_det).astype(int)
        normal = (y == 0)
        fpr = float(pred[normal].mean()) if normal.any() else float("nan")
        f1 = f1_score(y, pred, zero_division=0)
        pr = precision_score(y, pred, zero_division=0)
        rc = recall_score(y, pred, zero_division=0)
        rows.append(dict(route=name, n=nn, n_anom=int(y.sum()),
                         corridor_FPR=round(fpr, 4), F1=round(f1, 3),
                         Prec=round(pr, 3), Rec=round(rc, 3)))
        print(f"  {name:6s} n={nn:4d} anom={int(y.sum()):4d}  "
              f"FPR(normal)={fpr:.3f}  F1={f1:.3f}  P={pr:.3f}  R={rc:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(X.OUT / "AIC1_indep_reference.csv", index=False)
    print(f"\n  mean F1 across routes = {df.F1.mean():.3f}   [Paper-1 within-route F1=0.920, LORO=0.791]")
    print(f"  saved -> {X.OUT/'AIC1_indep_reference.csv'}")
    print("\nNOTE: PR-AUC/ROC-AUC are threshold-free and unchanged by the reference switch;")
    print("      only the operating point (tau) and thresholded metrics move.")


if __name__ == "__main__":
    main()
