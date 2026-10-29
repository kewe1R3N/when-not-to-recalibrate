#!/usr/bin/env python3
"""
M-decisive: label-free CODEBOOK re-estimation on a target-season corridor ride.
Freeze encoder + decoder; re-fit only the VQ codebook (k-means warm-started from
the trained codebook) on the encoder outputs of the target season's N St corridor.
Then recompute s_t on ALL routes of that season and measure rho(s_t, VDV).

Question: does corridor-only codebook adaptation RESTORE the learned score's link
to physical roughness (which season shift degraded, M1c)?  Corridor -> other routes
is the real transfer test.
"""
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
import aic2_exp as X

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMPS = ["spring", "summer", "autumn", "winter"]
ROUTES = ["11th", "13th", "16th", "Nst", "billy"]


def windows(data):
    n = (len(data) - X.WIN) // X.STEP + 1
    return n


def encode_ze(model, data, standardize=True):
    """Frozen-encoder latents z_e for all windows -> [N*T, latent].
    Input z-scored at model entry (Paper-1 inference convention, see X.zscore);
    standardize=False simulates the missing-standardization fault."""
    n = windows(data)
    data = X.zscore(data) if standardize else np.asarray(data, np.float32)
    chunks = []
    with torch.no_grad():
        for i in range(n):
            s = i * X.STEP
            w = data[s:s + X.WIN]
            if np.any(np.isnan(w)):
                continue
            x = torch.from_numpy(w).float().unsqueeze(0).to(dev)
            z = model.encoder(x)                       # [1, T, latent]
            chunks.append(z.reshape(-1, z.shape[-1]).cpu().numpy())
    return np.concatenate(chunks, 0)


def refit_codebook(model, ze):
    """k-means (warm start from current codebook) on corridor latents."""
    K = model.quant.embed.weight.shape[0]
    init = model.quant.embed.weight.detach().cpu().numpy()
    km = MiniBatchKMeans(n_clusters=K, init=init, n_init=1, max_iter=100,
                         batch_size=4096, random_state=42)
    km.fit(ze)
    return torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(dev)


def st_selfnorm(E, D):
    En = (E - np.nanmin(E)) / (np.nanmax(E) - np.nanmin(E) + 1e-8)
    Dn = (D - np.nanmin(D)) / (np.nanmax(D) - np.nanmin(D) + 1e-8)
    return 0.5 * En + 0.5 * Dn


def proxies(data, n):
    V = np.full(n, np.nan)
    for i in range(n):
        s = i * X.STEP
        w = data[s:s + X.WIN]
        V[i] = X.A.compute_window_proxies(w[:, 0], w[:, 1], w[:, 2],
                                          X.FS, X.A.config.HP_CUTOFF)["VDV_w"]
    return V


def rho_st_vdv(model, data):
    n = windows(data)
    E, D = X.raw_ED(model, data, dev)
    st = st_selfnorm(E, D)
    V = proxies(data, n)
    ok = ~(np.isnan(st) | np.isnan(V))
    return spearmanr(st[ok], V[ok]).correlation, int(ok.sum())


def main():
    m = X.A.load_model(X.CKPT, dev)
    orig = m.quant.embed.weight.detach().clone()
    rows = []
    for camp in CAMPS:
        corr = X.IPHONE / camp / "Nst.csv"
        if not corr.exists():
            continue
        cdata, _ = X.load_xyz(str(corr))
        # 1) frozen-codebook rho on every route of this season
        m.quant.embed.weight.data.copy_(orig)
        frozen = {}
        for r in ROUTES:
            f = X.IPHONE / camp / f"{r}.csv"
            if not f.exists():
                continue
            d, _ = X.load_xyz(str(f))
            if windows(d) < 30:
                continue
            rv, nn = rho_st_vdv(m, d)
            frozen[r] = (rv, nn)
        # 2) re-estimate codebook on THIS season's corridor
        ze = encode_ze(m, cdata)
        newcb = refit_codebook(m, ze)
        m.quant.embed.weight.data.copy_(newcb)
        for r in ROUTES:
            f = X.IPHONE / camp / f"{r}.csv"
            if not f.exists() or r not in frozen:
                continue
            d, _ = X.load_xyz(str(f))
            rv_a, nn = rho_st_vdv(m, d)
            rv_f, _ = frozen[r]
            rows.append(dict(camp=camp, route=r, n=nn,
                             rho_frozen=round(rv_f, 3), rho_cbadapt=round(rv_a, 3),
                             delta=round(rv_a - rv_f, 3)))
        m.quant.embed.weight.data.copy_(orig)   # restore
        print(f"[{camp}] corridor-adapted codebook applied to its routes:")
        for row in rows[-len(frozen):]:
            print(f"   {row['route']:5s} n={row['n']:4d}  frozen ρ={row['rho_frozen']:+.3f}"
                  f"  ->  cb-adapt ρ={row['rho_cbadapt']:+.3f}  (Δ={row['delta']:+.3f})")
    df = pd.DataFrame(rows)
    df.to_csv(X.OUT / "Mdec_codebook_adapt.csv", index=False)
    print("\n=== summary: mean Δρ(s_t,VDV) from corridor codebook adaptation ===")
    print(df.groupby("camp")["delta"].mean().round(3).to_string())
    print(f"overall mean Δ = {df['delta'].mean():+.3f}")
    print(f"saved -> {X.OUT/'Mdec_codebook_adapt.csv'}")


if __name__ == "__main__":
    main()
