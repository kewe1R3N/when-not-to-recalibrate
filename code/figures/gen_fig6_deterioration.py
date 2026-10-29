# Fig 6: corridor-DiD deterioration disentanglement — paired injected-minus-clean
# recovery + honest residual-scale framing (no "resolution floor" overclaim).
from paper_plot_style import *
import numpy as np
import pandas as pd

df = pd.read_csv(f"{RESULTS}/DET_synthetic_recovery.csv")
factor = df["factor"].values
D_paired = df["D_injloc_minus_clean"].values      # pure injection effect (paired)
D_clean = df["D_injloc_clean"].values             # same location's own background drift
D_ctrl = df["D_ctrl"].values                       # separate control location
RESID_SCALE = 0.04    # observed negative-control residual scale (not a "floor")

fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6))

# (a) paired recovery vs injected magnitude, alongside the background drift it corrects for
ax = axes[0]
ax.axhspan(-RESID_SCALE, RESID_SCALE, color=OI["grey"], alpha=0.18, lw=0)
ax.text(1.02, RESID_SCALE * 0.5, "observed residual\nscale ($\\pm0.04$)",
        fontsize=FONT_SIZE - 2, color="#555555", va="center")
ax.plot(factor, D_paired, "-o", color=OI["verm"], lw=1.4, markersize=5,
        label="paired injected$-$clean ($\\rho{=}{+}1.0$)")
ax.plot(factor, D_clean, "-^", color=OI["yellow"], lw=1.2, markersize=4,
        label="same location, clean (background drift)")
ax.plot(factor, D_ctrl, "-s", color=OI["blue"], lw=1.2, markersize=4,
        label="rest of route (aggregate)")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("Injected amplitude factor (chronological)")
ax.set_ylabel("Condition-index difference")
ax.legend(frameon=False, loc="upper left", fontsize=FONT_SIZE - 2)
ax.text(-0.16, 1.04, "(a)", transform=ax.transAxes, va="top", fontweight="bold")

# (b) paired recovered signal vs the observed (not "floor") residual scale
ax = axes[1]
recovered = float(D_paired[-1])
bars = ax.bar([0, 1], [recovered, RESID_SCALE], width=0.55,
              color=[OI["verm"], OI["grey"]], edgecolor="black", lw=0.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["paired injection\nresponse", "observed\nresidual scale"])
ax.set_ylabel("Condition-index change")
ax.text(0, recovered + 0.008, f"{recovered:.2f}", ha="center", fontsize=FONT_SIZE - 1)
ax.text(1, RESID_SCALE + 0.008, f"{RESID_SCALE:.2f}", ha="center", fontsize=FONT_SIZE - 1)
ax.set_ylim(0, max(recovered, RESID_SCALE) * 1.25)
ax.text(-0.16, 1.04, "(b)", transform=ax.transAxes, va="top", fontweight="bold")

fig.tight_layout()
save_fig(fig, "fig6_deterioration")
