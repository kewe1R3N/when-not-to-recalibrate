#!/usr/bin/env python3
"""
R008 + codebook ablation — the two reviewer-critical checks:

(A) DETECTION-UTILITY LANE (labeled loro folds 1/3/4; fold 2 excluded, Paper-1
    itself reports NaN there, n_test=431):
    Does corridor codebook re-estimation PRESERVE anomaly detection (PR-AUC /
    ROC-AUC on ground-truth window labels)?  Deployment-analog protocol:
    refit the codebook ONCE on the unlabeled N St corridor ride (fold_4 test
    data, labels never touched), then evaluate on all labeled folds.
    Note: threshold-only recalibration provably cannot change PR-AUC (same
    scores, different tau) — verified analytically, not re-computed.

(B) ABLATION CONTROLS (season contexts, label-free rho(s_t, VDV)):
    Is the M-decisive rho recovery more than a trivial nearest-neighbor-novelty
    effect?  Controls:
      cb-warm   : k-means warm-started from the trained codebook (ours, M-dec)
      cb-random : k-means++ from scratch on the same corridor latents
      latent-kNN: per-window mean 1-NN distance to corridor latents
                  (the "trivial" distance detector, no codebook/decoder at all)

Scoring conventions match prior blocks: Part A uses smoothed self-normalized
s_t (detection pipeline); Part B uses unsmoothed s_t like CB.rho_st_vdv (M-dec).
"""
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, roc_auc_score
import aic2_exp as X
import aic2_codebook as CB

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LORO = X.CODE / "latex_output" / "loro_results"
FOLDS = {"13th": ("fold_1_13th", 0.5997), "billy": ("fold_3_Billy", 0.6018),
         "Nst": ("fold_4_N", 0.4532)}          # Paper-1 PR-AUC for context only
CAMPS = ["spring", "summer", "autumn", "winter"]
ROUTES = ["11th", "13th", "16th", "Nst", "billy"]


def window_latents(model, data, n):
    """Per-window frozen-encoder latents, list of [T, D] (window-aligned).
    Input z-scored at model entry (Paper-1 inference convention)."""
    data = X.zscore(data)
    zs = []
    with torch.no_grad():
        for i in range(n):
            w = data[i * X.STEP: i * X.STEP + X.WIN]
            z = model.encoder(torch.from_numpy(w).float().unsqueeze(0).to(dev))
            zs.append(z.reshape(-1, z.shape[-1]).cpu().numpy())
    return zs


def refit_codebook_random(model, ze):
    """Control: k-means++ from scratch (NOT warm-started) on corridor latents."""
    K = model.quant.embed.weight.shape[0]
    km = MiniBatchKMeans(n_clusters=K, init="k-means++", n_init=1, max_iter=100,
                         batch_size=4096, random_state=0)
    km.fit(ze)
    return torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(dev)


def knn_window_scores(nn_index, zs):
    """Trivial baseline: mean 1-NN distance of a window's latents to corridor latents."""
    out = np.full(len(zs), np.nan)
    for i, z in enumerate(zs):
        d, _ = nn_index.kneighbors(z, n_neighbors=1)
        out[i] = float(d.mean())
    return out


def st_smoothed(model, data):
    E, D = X.raw_ED(model, data, dev)
    return X.smooth(CB.st_selfnorm(E, D))


# ---------------------------------------------------------------- Part A
def part_A(m, orig_cb):
    print("=== (A) Detection-utility lane: PR-AUC on labeled loro folds ===")
    # corridor for the deployment-analog refit = a DIFFERENT N St ride (the
    # Lincoln summer iPhone corridor), not the fold_4_N test.csv being scored
    # below — fitting the codebook on the very windows whose PR-AUC is then
    # reported (as the original fold_4_N-on-fold_4_N choice did) leaks the
    # "Nst" evaluation fold into its own codebook fit.
    corridor_data, _ = X.load_xyz(str(X.IPHONE / "summer" / "Nst.csv"))
    n_corr = CB.windows(corridor_data)
    ze_corr = CB.encode_ze(m, corridor_data)
    print(f"  corridor (N St, unlabeled use): {n_corr} windows, {len(ze_corr)} latents")
    nn_index = NearestNeighbors(n_neighbors=1).fit(ze_corr)
    cb_warm = CB.refit_codebook(m, ze_corr)
    cb_rand = refit_codebook_random(m, ze_corr)

    rows = []
    for name, (fold, p1_ref) in FOLDS.items():
        data, anom = X.load_xyz(str(LORO / fold / "data" / "test.csv"))
        n = CB.windows(data)
        y = (~X.window_normal_mask(anom, n)).astype(int)     # 1 = anomalous window
        zs = window_latents(m, data, n)                      # encoder is frozen everywhere

        scores = {}
        m.quant.embed.weight.data.copy_(orig_cb)
        scores["frozen"] = st_smoothed(m, data)
        m.quant.embed.weight.data.copy_(cb_warm)
        scores["cb-warm (ours)"] = st_smoothed(m, data)
        m.quant.embed.weight.data.copy_(cb_rand)
        scores["cb-random"] = st_smoothed(m, data)
        m.quant.embed.weight.data.copy_(orig_cb)
        scores["latent-kNN"] = X.smooth(knn_window_scores(nn_index, zs))

        for meth, s in scores.items():
            ok = ~np.isnan(s)
            ap = average_precision_score(y[ok], s[ok])
            roc = roc_auc_score(y[ok], s[ok])
            rows.append(dict(fold=name, method=meth, PR_AUC=round(ap, 4),
                             ROC_AUC=round(roc, 4), n_windows=int(ok.sum()),
                             n_anom=int(y[ok].sum()), paper1_PR_AUC=p1_ref))
        print(f"  [{name}] n={n} anom={int(y.sum())}  " +
              "  ".join(f"{k}: AP={rows[-4:][i]['PR_AUC']:.3f}"
                        for i, k in enumerate(scores)))
    df = pd.DataFrame(rows)
    df.to_csv(X.OUT / "R008_prauc_lane.csv", index=False)
    print("\n  PR-AUC pivot (rows=method, cols=fold):")
    print(df.pivot_table(index="method", columns="fold", values="PR_AUC").round(3).to_string())
    print("  ROC-AUC pivot:")
    print(df.pivot_table(index="method", columns="fold", values="ROC_AUC").round(3).to_string())
    print(f"  -> {X.OUT/'R008_prauc_lane.csv'}")


# ---------------------------------------------------------------- Part B
def part_B(m, orig_cb):
    print("\n=== (B) Codebook ablation: warm vs random vs latent-kNN (rho vs VDV) ===")
    rows = []
    for camp in CAMPS:
        corr = X.IPHONE / camp / "Nst.csv"
        if not corr.exists():
            continue
        cdata, _ = X.load_xyz(str(corr))
        m.quant.embed.weight.data.copy_(orig_cb)
        ze = CB.encode_ze(m, cdata)
        nn_index = NearestNeighbors(n_neighbors=1).fit(ze)
        cb_warm = CB.refit_codebook(m, ze)
        cb_rand = refit_codebook_random(m, ze)

        for r in ROUTES:
            f = X.IPHONE / camp / f"{r}.csv"
            if not f.exists():
                continue
            d, _ = X.load_xyz(str(f))
            n = CB.windows(d)
            if n < 30:
                continue
            V = CB.proxies(d, n)
            ok0 = ~np.isnan(V)

            def rho_with(cb):
                m.quant.embed.weight.data.copy_(cb)
                E, D = X.raw_ED(m, d, dev)
                st = CB.st_selfnorm(E, D)
                ok = ok0 & ~np.isnan(st)
                return spearmanr(st[ok], V[ok]).correlation

            rho_frozen = rho_with(orig_cb)
            rho_warm = rho_with(cb_warm)
            rho_rand = rho_with(cb_rand)
            m.quant.embed.weight.data.copy_(orig_cb)
            zs = window_latents(m, d, n)
            knn = knn_window_scores(nn_index, zs)
            ok = ok0 & ~np.isnan(knn)
            rho_knn = spearmanr(knn[ok], V[ok]).correlation

            rows.append(dict(camp=camp, route=r, n=n,
                             rho_frozen=round(rho_frozen, 3),
                             rho_cb_warm=round(rho_warm, 3),
                             rho_cb_random=round(rho_rand, 3),
                             rho_latent_knn=round(rho_knn, 3)))
            print(f"  [{camp}/{r}] frozen {rho_frozen:+.3f}  warm {rho_warm:+.3f}  "
                  f"random {rho_rand:+.3f}  kNN {rho_knn:+.3f}")
        m.quant.embed.weight.data.copy_(orig_cb)
    df = pd.DataFrame(rows)
    df.to_csv(X.OUT / "R008_ablation_rho.csv", index=False)
    print("\n  mean rho per method:")
    print(df[["rho_frozen", "rho_cb_warm", "rho_cb_random", "rho_latent_knn"]]
          .mean().round(3).to_string())
    print("  mean rho per method, worst season (spring):")
    sp = df[df.camp == "spring"]
    print(sp[["rho_frozen", "rho_cb_warm", "rho_cb_random", "rho_latent_knn"]]
          .mean().round(3).to_string())
    print(f"  -> {X.OUT/'R008_ablation_rho.csv'}")


def main():
    m = X.A.load_model(X.CKPT, dev)
    orig_cb = m.quant.embed.weight.detach().clone()
    part_A(m, orig_cb)
    part_B(m, orig_cb)


if __name__ == "__main__":
    main()
