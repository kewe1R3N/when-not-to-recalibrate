# Fig 3: recalibration centerpiece — (a) calibration error by method x context,
#        (b) frozen vs conformal corridor FPR with moving-block bootstrap 95% CIs
from paper_plot_style import *
import numpy as np
import pandas as pd
import re

df = pd.read_csv(f"{RESULTS}/B23_recal_table.csv")
ci = pd.read_csv(f"{RESULTS}/B_bootstrap_ci_v2.csv")
control=pd.read_csv(f"{RESULTS}/BASE_physical_calibration.csv")
control["method"]="V4+conformal"
df=pd.concat([df,control],ignore_index=True)
CONTEXTS = ["autumn", "winter", "spring", "android"]      # chronological
CTX_LAB = ["Oct 2023", "Nov–Dec 2023", "Jan 2024", "Android"]
METHODS = ["frozen", "z-score", "MAD", "conformal", "V4+conformal"]
M_LAB = ["Frozen", "z-score", "MAD", "Conformal", "$V_4$+conf."]
M_COL = [OI["grey"], OI["skyblue"], OI["yellow"], OI["blue"], OI["green"], OI["purple"]]

fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.8))

# (a) grouped bars: |FPR - 1%|
ax = axes[0]
w = 0.15
xs = np.arange(len(CONTEXTS))
for k, (mth, lab, col) in enumerate(zip(METHODS, M_LAB, M_COL)):
    vals = [float(df[(df.context == c) & (df.method == mth)]["calib_err"].iloc[0])
            for c in CONTEXTS]
    ax.bar(xs + (k - 2) * w, [100*v for v in vals], width=w, label=lab, color=col,
           edgecolor="black", linewidth=0.4)
ax.set_ylim(0, 15.5)
ax.set_xticks(xs)
ax.set_xticklabels(CTX_LAB)
ax.set_ylabel(r"Target deviation $|\mathrm{CER}-1\%|$ (pp)")
ax.set_xlabel("Shifted context")
ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.2,
          columnspacing=0.9)
ax.text(-0.14, 1.04, "(a)", transform=ax.transAxes, va="top", fontweight="bold")

# (b) FPR with bootstrap CIs: frozen vs conformal
ax = axes[1]
parse = lambda s: [float(v) for v in re.findall(r"[\d.]+", s)]
xs = np.arange(len(CONTEXTS))
for k, (col, lab, fcol, ccol) in enumerate(
        [(OI["grey"], "Frozen", "frozen_FPR", "frozen_CI"),
         (OI["blue"], "Conformal", "conformal_FPR", "conformal_CI")]):
    fpr = [100*float(ci[ci.context == c][fcol].iloc[0]) for c in CONTEXTS]
    los, his = zip(*[[100*u for u in parse(ci[ci.context == c][ccol].iloc[0])] for c in CONTEXTS])
    err = [np.array(fpr) - np.array(los), np.array(his) - np.array(fpr)]
    ax.errorbar(xs + (k - 0.5) * 0.18, fpr, yerr=err, fmt="o", color=col,
                capsize=3, markersize=5, label=lab, lw=1.2)
ax.axhline(1.0, color=OI["verm"], lw=1.2, ls="--")
ax.text(-0.45, 1.25, "1% target", color=OI["verm"], fontsize=FONT_SIZE - 2,
        ha="left", va="bottom")
ax.set_xticks(xs)
ax.set_xticklabels(CTX_LAB)
ax.set_xlim(-0.5, len(CONTEXTS) - 0.5)
ax.set_ylabel("Held-out CER (%)")
ax.set_xlabel("Shifted context")
ax.legend(frameon=False, loc="upper left")
ax.text(-0.16, 1.04, "(b)", transform=ax.transAxes, va="top", fontweight="bold")

fig.tight_layout()
save_fig(fig, "fig3_recalibration")
