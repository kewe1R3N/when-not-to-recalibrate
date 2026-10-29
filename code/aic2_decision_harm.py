#!/usr/bin/env python3
"""Decision quality by size and by cluster (written after the decision-split results were known; descriptive).
G(pi): mean reduction in |CER_B - alpha| relative to never updating; accepted harm, avoided harm, forgone benefit
(sums over units, pp); unit-, stretch- and street-cluster-weighted means; cluster bootstrap of mean differences.
Clusters: evaluation stretch = (evaluation pass, street); street cluster = physical street (Lincoln: one per ride)."""
import numpy as np, pandas as pd
import aic2_exp as X

o = pd.read_csv(X.OUT / "DS_decision_split.csv").rename(columns={"context": "direction"})
ev_o = {"Oct": "L-Oct", "Nov-Dec": "L-NovDec", "Jan": "L-Jan", "Android": "L-Android",
        "6->16": "i16@East Bay Street", "16->6": "i6@East Bay Street", "3->12": "i12@East Bay Street",
        "12->3": "i3@East Bay Street", "Main R2a->R2b": "R2b@Main", "Main R2b->R2a": "R2a@Main"}
o["stretch"] = o.direction.map(ev_o); o["set"] = "original"
n = pd.read_csv(X.OUT / "CA_decision_split.csv"); n = n[n.status == "split"].copy()
n["stretch"] = n.direction.str.split("->").str[1] + "@" + n.street; n["set"] = "android"
d = pd.concat([o, n], ignore_index=True)
d["street"] = d.stretch.str.split("@").str[1]
d.loc[d.stretch.str.startswith("L-"), "street"] = d.stretch      # Lincoln: one cluster per target ride
d0, d1 = d.dB_frozen, d.dB_cand
acc = {"Never": pd.Series(False, index=d.index), "Always": pd.Series(True, index=d.index),
       "CER-only": d.CERonly_accepts.astype(bool), "Full": d.Full_accepts.astype(bool), "Oracle": d1 < d0}
rows = []
for sub, m in [("original 10", d.set == "original"), ("all 60", d.index >= 0)]:
    x = d[m]
    for k, a in acc.items():
        a = a[m]; dv = np.where(a, x.dB_cand, x.dB_frozen)
        gain = x.dB_frozen - dv
        r = dict(set=sub, strategy=k, mean_dev=round(dv.mean(), 2), G=round(gain.mean(), 2),
                 harm_accepted=round(float(np.clip(x.dB_cand - x.dB_frozen, 0, None)[a].sum()), 2),
                 harm_avoided=round(float(np.clip(x.dB_cand - x.dB_frozen, 0, None)[~a].sum()), 2),
                 benefit_realized=round(float(np.clip(x.dB_frozen - x.dB_cand, 0, None)[a].sum()), 2),
                 benefit_forgone=round(float(np.clip(x.dB_frozen - x.dB_cand, 0, None)[~a].sum()), 2))
        for lvl in ("stretch", "street"):
            r[f"mean_dev_{lvl}_w"] = round(pd.Series(dv, index=x.index).groupby(x[lvl]).mean().mean(), 2)
        rows.append(r)
out = pd.DataFrame(rows); out.to_csv(X.OUT / "DH_decision_harm.csv", index=False)
print(out.to_string(index=False))
print("clusters (all 60): stretches", d.stretch.nunique(), " streets/rides", d.street.nunique())
# cluster bootstrap of mean-deviation differences (all 60), resampling streets
rng = np.random.default_rng(0); B = 5000
dev = {k: pd.Series(np.where(a, d.dB_cand, d.dB_frozen), index=d.index) for k, a in acc.items()}
cl = d.street.values; ucl = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in ucl}
res = []
for lvl, groups in [("street", cl), ("stretch", d.stretch.values)]:
    u = np.unique(groups); ix = {c: np.where(groups == c)[0] for c in u}
    for a_, b_ in [("CER-only", "Always"), ("Full", "Always"), ("CER-only", "Never"), ("Full", "Never")]:
        diff = (dev[a_] - dev[b_]).values; bs = []
        for _ in range(B):
            s = rng.choice(u, len(u)); sel = np.concatenate([ix[c] for c in s]); bs.append(diff[sel].mean())
        res.append(dict(cluster=lvl, n_clusters=len(u), contrast=f"{a_} - {b_}", mean=round(diff.mean(), 2),
                        lo=round(np.percentile(bs, 2.5), 2), hi=round(np.percentile(bs, 97.5), 2)))
r = pd.DataFrame(res); r.to_csv(X.OUT / "DH_cluster_bootstrap.csv", index=False); print(r.to_string(index=False))
# per-cluster contributions and leave-one-cluster-out ratios (forgone benefit / avoided harm)
pc = []
for lab, a in [("CER-only", acc["CER-only"]), ("Full", acc["Full"])]:
    d["avoid_" + lab] = np.where(~a, np.clip(d.dB_cand - d.dB_frozen, 0, None), 0)
    d["forg_" + lab] = np.where(~a, np.clip(d.dB_frozen - d.dB_cand, 0, None), 0)
g = d.groupby("street")[["avoid_CER-only", "forg_CER-only", "avoid_Full", "forg_Full"]].sum()
g["units"] = d.groupby("street").size()
for c in g.index:
    h = g.drop(c).sum()
    g.loc[c, "loo_ratio_CER-only"] = h["forg_CER-only"] / h["avoid_CER-only"] if h["avoid_CER-only"] > 0 else np.inf
    g.loc[c, "loo_ratio_Full"] = h["forg_Full"] / h["avoid_Full"] if h["avoid_Full"] > 0 else np.inf
g = g.round(2).sort_values("forg_Full", ascending=False); g.to_csv(X.OUT / "DH_cluster_contrib.csv"); print(g.to_string())
