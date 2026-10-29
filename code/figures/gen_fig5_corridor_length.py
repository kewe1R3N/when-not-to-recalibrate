# Fig 5: corridor-length curve — held-out corridor FPR vs #calibration windows
from paper_plot_style import *
import numpy as np
import pandas as pd

df = pd.read_csv(f"{RESULTS}/R010_corridor_length.csv")
CTX = {"summer": ("May 2024", OI["blue"]), "spring": ("Jan 2024", OI["verm"]),
       "autumn": ("Oct 2023", OI["green"]), "winter": ("Nov–Dec 2023", OI["orange"]),
       "android": ("Android", OI["purple"])}

fig, ax = plt.subplots(figsize=(COL_W, 2.5))
for ctx, (lab, col) in CTX.items():
    d = df[df.context == ctx].sort_values("n_cal")
    if d.empty:
        continue
    ax.plot(d.n_cal, 100*d.FPR_mean, "-o", color=col, label=lab, markersize=3.5, lw=1.3)
    ax.fill_between(d.n_cal, 100*d.FPR_q05, 100*d.FPR_q95, color=col, alpha=0.12, lw=0)
ax.axhline(1.0, color="black", lw=1.0, ls="--")
ax.text(60, 1.6, "1% target", fontsize=FONT_SIZE - 2, ha="center")
ax.axvspan(0, 99, color=OI["grey"], alpha=0.15, lw=0)
ax.text(6, 40, "no finite\nthreshold\n($n_{\\mathrm{cal}}{<}99$)", fontsize=FONT_SIZE - 2, color="#555555")
ax.axvline(300, color=OI["grey"], lw=1.0, ls=":")
ax.text(308, 20, "provisional\nmargin\n300 windows", fontsize=FONT_SIZE - 2, color="#555555")
ax.set_xlabel("Corridor calibration windows $n_{\\mathrm{cal}}$")
ax.set_ylabel("Held-out CER (%)")
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=FONT_SIZE - 2, ncol=1, loc="upper right",
          bbox_to_anchor=(1.0, 1.0))
save_fig(fig, "fig5_corridor_length")
