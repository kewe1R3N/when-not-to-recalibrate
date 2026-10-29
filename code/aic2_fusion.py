#!/usr/bin/env python3
"""
Path-1 complementarity check: does fusing the learned score with a physical
energy score beat either alone on labeled detection?  Strictly cross-fold:
the fusion weight w is selected on the OTHER two folds (by mean AP) and
evaluated on the held-out fold — labels of the evaluation fold are never used
for tuning.  Scores are rank-normalized per fold (scale-free, label-free).

Also reports the E-only / D-only components of s_t (diagnostic) and the rank
correlation between s_t and RMS (how much complementary information exists).
"""
import numpy as np, pandas as pd, torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score
import aic2_exp as X
import aic2_codebook as CB

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LORO = X.CODE / "latex_output" / "loro_results"
FOLDS = {"13th": "fold_1_13th", "billy": "fold_3_Billy", "Nst": "fold_4_N"}
WGRID = np.round(np.arange(0, 1.01, 0.1), 2)


def rank01(s):
    out = np.full_like(s, np.nan, dtype=float)
    ok = ~np.isnan(s)
    out[ok] = rankdata(s[ok]) / ok.sum()
    return out


def main():
    m = X.A.load_model(X.CKPT, dev)
    F = {}
    for name, fold in FOLDS.items():
        data, anom = X.load_xyz(str(LORO / fold / "data" / "test.csv"))
        n = CB.windows(data)
        y = (~X.window_normal_mask(anom, n)).astype(int)
        E, D = X.raw_ED(m, data, dev)
        En = (E - np.nanmin(E)) / (np.nanmax(E) - np.nanmin(E) + 1e-8)
        Dn = (D - np.nanmin(D)) / (np.nanmax(D) - np.nanmin(D) + 1e-8)
        rms = np.array([float(np.sqrt(np.mean(np.linalg.norm(
            data[i*X.STEP:i*X.STEP+X.WIN], axis=1) ** 2))) for i in range(n)])
        F[name] = dict(
            y=y,
            st=rank01(X.smooth(0.5 * En + 0.5 * Dn)),
            E=rank01(X.smooth(En)), D=rank01(X.smooth(Dn)),
            rms=rank01(X.smooth(rms)),
        )

    def ap(fold, s):
        y = F[fold]["y"]; ok = ~np.isnan(s)
        return average_precision_score(y[ok], s[ok])

    print("=== components (diagnostic) ===")
    rows = []
    for f in F:
        r = {k: round(ap(f, F[f][k]), 4) for k in ("st", "E", "D", "rms")}
        rho = spearmanr(F[f]["st"][~np.isnan(F[f]["st"])],
                        F[f]["rms"][~np.isnan(F[f]["rms"])]).correlation
        rows.append(dict(fold=f, **r, rank_corr_st_rms=round(rho, 3)))
        print(f"  [{f}] s_t {r['st']:.3f}  E-only {r['E']:.3f}  D-only {r['D']:.3f}  "
              f"RMS {r['rms']:.3f}  rank-corr(s_t,RMS)={rho:.3f}")
    pd.DataFrame(rows).to_csv(X.OUT / "FUSION_components.csv", index=False)

    print("\n=== cross-fold fusion: w*rank(s_t) + (1-w)*rank(RMS) ===")
    frows = []
    for hold in F:
        others = [f for f in F if f != hold]
        mean_ap = {w: np.mean([ap(f, w * F[f]["st"] + (1 - w) * F[f]["rms"])
                               for f in others]) for w in WGRID}
        wstar = max(mean_ap, key=mean_ap.get)
        fused = wstar * F[hold]["st"] + (1 - wstar) * F[hold]["rms"]
        frows.append(dict(hold_out=hold, w_star=wstar,
                          AP_fused=round(ap(hold, fused), 4),
                          AP_st=round(ap(hold, F[hold]["st"]), 4),
                          AP_rms=round(ap(hold, F[hold]["rms"]), 4),
                          AP_fused_w05=round(ap(hold, 0.5 * F[hold]["st"] +
                                                0.5 * F[hold]["rms"]), 4)))
        print(f"  hold-out {hold}: w*={wstar:.1f} (picked on {others})  "
              f"fused AP {frows[-1]['AP_fused']:.3f}  vs s_t {frows[-1]['AP_st']:.3f}  "
              f"RMS {frows[-1]['AP_rms']:.3f}  (w=0.5 ref {frows[-1]['AP_fused_w05']:.3f})")
    df = pd.DataFrame(frows)
    df.to_csv(X.OUT / "FUSION_crossfold.csv", index=False)
    print(f"\n  mean: fused {df.AP_fused.mean():.4f}  s_t {df.AP_st.mean():.4f}  "
          f"RMS {df.AP_rms.mean():.4f}  fused(w=0.5) {df.AP_fused_w05.mean():.4f}")
    print(f"saved -> FUSION_components.csv, FUSION_crossfold.csv")


if __name__ == "__main__":
    main()
