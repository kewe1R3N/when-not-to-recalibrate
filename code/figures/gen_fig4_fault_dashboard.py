# Fig 4: fault-injection dashboard — three label-free corridor signals per fault:
#   (a) two-sided FPR budget check, (b) physical-proxy health rho, (c) codebook-usage JS
from paper_plot_style import *
import numpy as np
import pandas as pd

df = pd.read_csv(f"{RESULTS}/FI_fault_grid.csv")
jsctx = pd.read_csv(f"{RESULTS}/FI_js_context.csv")
floor = float(jsctx[jsctx.kind == "floor"]["JS"].iloc[0])
env_lo = float(jsctx[jsctx.kind.isin(["season", "device"])]["JS"].min())
env_hi = float(jsctx[jsctx.kind.isin(["season", "device"])]["JS"].max())

ORDER = ["none", "unit_ms2", "gravity_rm", "missing_std", "axswap_YZ",
         "axswap_XY", "rate_2x"]
LAB = ["Clean", "Unit m/s$^2$", "Gravity-rem.", "Missing std.",
       "Swap Y$\\leftrightarrow$Z", "Swap X$\\leftrightarrow$Y",
       "Rate 2$\\times$"]


def style_xticks(ax, xs, labels):
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE - 2, rotation=35, ha="right",
                       rotation_mode="anchor")
CLS_COL = {"-": OI["grey"], "A (affine)": OI["blue"], "B (stack)": OI["verm"],
           "B (orientation)": OI["orange"], "B (temporal)": OI["purple"]}
df = df.set_index("fault").loc[ORDER].reset_index()
cols = [CLS_COL[c] for c in df["cls"]]
xs = np.arange(len(df))
clean_fpr = 100*float(df[df.fault == "none"]["FPR_frozen"].iloc[0])

fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.35), constrained_layout=True)

# (a) corridor FPR (two-sided check)
ax = axes[0]
ax.bar(xs, 100*df["FPR_frozen"], color=cols, edgecolor="black", linewidth=0.4)
ax.axhline(clean_fpr, color=OI["grey"], lw=1.0, ls=":")
ax.text(-0.35, clean_fpr + 0.15, "clean level", ha="left",
        fontsize=FONT_SIZE - 2, color="#555555")
ax.axhline(1.0, color=OI["verm"], lw=1.0, ls="--")
ax.text(-0.35, 1.12, "1% target", ha="left",
        fontsize=FONT_SIZE - 2, color=OI["verm"])
for i, v in enumerate(df["FPR_frozen"]):
    if v == 0:
        ax.text(i, 1.35, "0 observed", rotation=90, ha="center", va="bottom",
                fontsize=FONT_SIZE - 2, color=OI["verm"], fontweight="bold")
ax.set_ylabel("CER, frozen threshold (%)")
style_xticks(ax, xs, LAB)
ax.text(-0.24, 1.10, "(a)", transform=ax.transAxes, va="top", fontweight="bold")

# (b) physical-proxy health check
ax = axes[1]
ax.bar(xs, df["rho_corridor"], color=cols, edgecolor="black", linewidth=0.4)
ax.axhline(float(df[df.fault == "none"]["rho_corridor"].iloc[0]),
           color=OI["grey"], lw=1.0, ls=":")
ax.set_ylabel(r"Corridor $\rho\,(s_t^{\mathrm{aux}},\ V_4)$")
ax.set_ylim(0, 1.0)
style_xticks(ax, xs, LAB)
ax.text(-0.24, 1.10, "(b)", transform=ax.transAxes, va="top", fontweight="bold")

# (c) codebook-usage JS divergence (log scale, exact zeros clipped + annotated)
ax = axes[2]
CLIP = 2e-4
vals = np.maximum(df["JS_usage"].values, CLIP)
ax.bar(xs, vals, color=cols, edgecolor="black", linewidth=0.4)
ax.set_yscale("log")
ax.set_ylim(CLIP, 0.6)
ax.axhspan(env_lo, env_hi, color=OI["green"], alpha=0.18, lw=0)
ax.text(-0.35, np.sqrt(env_lo * env_hi), "observed context\nrange",
        fontsize=FONT_SIZE - 2, color=OI["green"], va="center", ha="left")
ax.axhline(floor, color=OI["grey"], lw=1.0, ls=":")
ax.text(-0.35, floor * 0.72, "within-ride ref.", ha="left",
        fontsize=FONT_SIZE - 2, color="#555555")
for i, v in enumerate(df["JS_usage"]):
    if v == 0:
        ax.text(i, CLIP * 1.3, "0 (exact)", rotation=90, ha="center",
                va="bottom", fontsize=FONT_SIZE - 2, color=OI["blue"],
                fontweight="bold")
ax.set_ylabel("Codebook-usage JS divergence")
style_xticks(ax, xs, LAB)
ax.text(-0.24, 1.10, "(c)", transform=ax.transAxes, va="top", fontweight="bold")

# class legend
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=OI["grey"], label="Clean"),
           mpatches.Patch(color=OI["blue"], label="Class A (affine)"),
           mpatches.Patch(color=OI["orange"], label="Class B (orientation)"),
           mpatches.Patch(color=OI["verm"], label="Class B (stack)"),
           mpatches.Patch(color=OI["purple"], label="Class B (temporal)")]
fig.legend(handles=handles, ncol=5, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, 1.10))
save_fig(fig, "fig4_fault_dashboard")
