#!/usr/bin/env python3
"""POST HOC sensitivity (after the decision-split and Citadel Android results were known):
(1) tie handling: keep the incumbent only on a strict refusal (candidate deviation above the incumbent's on the
    decision half), so ties go to the candidate; (2) JS limit rounded up to 0.045. Uses DS_/CA_ result tables only."""
import numpy as np, pandas as pd
import aic2_exp as X

A = X.ALPHA * 100
o = pd.read_csv(X.OUT / "DS_decision_split.csv").rename(columns={"context": "direction"}).assign(set="original")
n = pd.read_csv(X.OUT / "CA_decision_split.csv"); n = n[n.status == "split"].assign(set="android")
pr = pd.read_csv(X.OUT / "CA_prompts.csv").set_index("pass_")
d = pd.concat([o, n], ignore_index=True)
dev = lambda c: (d[c] * 100 - A).abs()
dA0, dA1, dB0, dB1 = dev("CER_A_frozen"), dev("CER_A_cand"), d.dB_frozen, d.dB_cand
strict = dA1 <= dA0                                   # accept unless strictly worse on A
def rho_js(r):
    if r.set == "original":
        return {"Oct": (0.913, 0.0165), "Nov-Dec": (0.959, 0.0441), "Jan": (0.924, 0.0386), "Android": (0.829, 0.0394)}.get(
            r.direction, (0.95 if "R2" not in r.direction else (0.867 if "R2a->" in r.direction else 0.815),
                          0.0137 if "R2" not in r.direction else (0.0507 if "R2a->" in r.direction else 0.0573)))
    c = r.direction.split("->")[0]
    return (pr.loc[c, "rho_aux"], pr.loc[c, "JS"]) if c in pr.index else (1.0, 0.0137)
RJ = np.array([rho_js(r) for r in d.itertuples()])
hold45 = (RJ[:, 0] < 0.80) | (RJ[:, 1] > 0.045)
rules = {"Always": pd.Series(True, index=d.index), "CER-only (primary)": d.CERonly_accepts.astype(bool),
         "Full (primary)": d.Full_accepts.astype(bool), "CER-only, strict refusal": strict,
         "Full, strict refusal": strict & ~d.hold.astype(bool), "Full, JS limit 0.045": d.CERonly_accepts.astype(bool) & ~hold45}
u = d.update_on_B; rows = []
for sub, m in [("original 10", d.set == "original"), ("all 60", d.index >= 0)]:
    for k, acc in rules.items():
        a = acc[m]; uu = u[m]
        rows.append(dict(set=sub, rule=k, mean_dev_B=round(float(np.where(a, dB1[m], dB0[m]).mean()), 2),
                         harmful_accepted=int((a & (uu == "harmful")).sum()), harmful=int((uu == "harmful").sum()),
                         beneficial_kept=int((~a & (uu == "beneficial")).sum()), beneficial=int((uu == "beneficial").sum())))
out = pd.DataFrame(rows); out.to_csv(X.OUT / "TS_tie_sensitivity.csv", index=False); print(out.to_string(index=False))
# distinct evaluation stretches among the 50 Android-extension directions
ev = n.direction.str.split("->").str[1] + "@" + n.street
print("distinct (evaluation pass, street) among 50:", ev.nunique())
print("Columbia R2b->R2a CER_A:", o.loc[o.direction.str.contains("R2b->R2a"), ["CER_A_frozen", "CER_A_cand"]].values)
