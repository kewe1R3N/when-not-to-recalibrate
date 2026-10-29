# Fig 1: leave-one-source-out corridor FPR drift matrix (× of the 1% target)
from paper_plot_style import *
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

df = pd.read_csv(f"{RESULTS}/M1v2_drift_matrix.csv")
df["eval"] = df["eval"].str.replace("(held-out half)", "", regex=False)
CAMPS = ["autumn", "winter", "spring", "summer"]          # chronological
DATE = {"autumn": "Oct 2023", "winter": "Nov–Dec 2023",
        "spring": "Jan 2024", "summer": "May 2024", "android": "Android"}
EVALS = CAMPS + ["android"]
piv = df.pivot_table(index="source", columns="eval", values="inflation_x")
piv = piv.reindex(index=CAMPS, columns=EVALS)

fig, ax = plt.subplots(figsize=(COL_W, 2.35))
M = piv.values.astype(float)
masked = np.ma.masked_invalid(M)
# Muted diverging map centered on the 1x budget: on-budget reads neutral (cream), silent
# (<1x, under-alarming) reads slate-blue, over-alarming (>1x) reads terracotta. Encodes the
# paper's BIDIRECTIONAL-drift claim, which a one-sided hot map (silent cells pale = look
# 'fine') hides.
CMAP = MUTED_DIV
norm = TwoSlopeNorm(vcenter=1.0, vmin=0, vmax=6)
im = ax.imshow(masked, cmap=CMAP, norm=norm)
style_heatmap(ax, M.shape[0], M.shape[1])
ax.set_xticks(range(len(EVALS)))
ax.set_xticklabels([DATE[c] for c in EVALS], fontsize=FONT_SIZE - 1,
                   rotation=22, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(CAMPS)))
ax.set_yticklabels([DATE[c] for c in CAMPS], fontsize=FONT_SIZE - 1)
ax.set_xlabel("Target ride (diagonal: source held-out block)")
ax.set_ylabel("Calibration source campaign")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        if np.isnan(v):
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#ececec", zorder=2))
            ax.text(j, i, "src", ha="center", va="center", color="#999999",
                    fontsize=FONT_SIZE - 2, style="italic", zorder=3)
        else:
            tc = cell_text_color(v, norm, CMAP)
            if v == 0:                                    # silent: 0x + small italic tag
                ax.text(j, i - 0.13, "0%", ha="center", va="center",
                        color=tc, fontsize=FONT_SIZE - 1, zorder=3)
                ax.text(j, i + 0.22, "zero", ha="center", va="center",
                        color=tc, fontsize=FONT_SIZE - 3, style="italic", zorder=3)
            else:
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        color=tc, fontsize=FONT_SIZE - 1, zorder=3)
for i in range(len(CAMPS)):
    ax.add_patch(plt.Rectangle((i-.49,i-.49),.98,.98,fill=False,edgecolor="black",lw=.8,zorder=4))
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
cb.set_label("CER (%); target 1%")
cb.ax.axhline(1.0, color="#333333", lw=1.0)           # the budget = diverging center
refine_colorbar(cb, ticks=[0, 1, 2, 4, 6])
save_fig(fig, "fig1_drift_matrix")
