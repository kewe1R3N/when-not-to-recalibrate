# Fig 2: ranking robustness grid — rho(s_t, VDV) across campaigns x routes (+ Android)
from paper_plot_style import *
import numpy as np
import pandas as pd

df = pd.read_csv(f"{RESULTS}/M1cv2_ranking_allroutes.csv")
CAMPS = ["autumn", "winter", "spring", "summer", "android"]   # chronological
ROUTES = ["11th", "13th", "16th", "Nst", "billy"]
piv = df.pivot_table(index="camp", columns="route", values="rho")
piv = piv.reindex(index=CAMPS, columns=ROUTES)

fig, ax = plt.subplots(figsize=(COL_W, 2.55))
M = piv.values.astype(float)
masked = np.ma.masked_invalid(M)
# Data live in ~0.79-0.96; a 0.5-1.0 scale flattens them all to near-identical dark green.
# Tighten to just below/above the data range so the (strong) contrasts are actually visible.
CMAP = MUTED_SEQ                                  # muted green, matches fig1's palette family
norm = plt.Normalize(vmin=0.75, vmax=0.98)
im = ax.imshow(masked, cmap=CMAP, norm=norm)
style_heatmap(ax, M.shape[0], M.shape[1])
ax.set_xticks(range(len(ROUTES)))
ax.set_xticklabels(["11th St", "13th St", "16th St", "N St\n(corridor)", "Billy\nWolff"])
ax.set_yticks(range(len(CAMPS)))
ax.set_yticklabels(["Oct 2023", "Nov–Dec 2023", "Jan 2024", "May 2024",
                    "Android"])
ax.set_xlabel("Route")
ax.set_ylabel("Context (campaign / device)")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        if np.isnan(v):
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#e5e5e5", zorder=2))
            ax.text(j, i, "n/a", ha="center", va="center", color="#888888",
                    fontsize=FONT_SIZE - 2, style="italic")
        else:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=cell_text_color(v, norm, CMAP), fontsize=FONT_SIZE - 1)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
cb.set_label(r"$\rho\,(s_t^{\mathrm{aux}},\ V_4)$")
refine_colorbar(cb, ticks=[0.75, 0.80, 0.85, 0.90, 0.95])
save_fig(fig, "fig2_ranking_grid")
