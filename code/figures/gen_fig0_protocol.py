# Fig 0: corridor-sentinel protocol overview (schematic, deterministic).
# Redrawn for a clean 4-column pipeline: Input -> Model entry -> Label-free signal
# -> Deployment action, with the muted palette shared with fig1/fig2.
from paper_plot_style import *
import matplotlib.patches as mp
from matplotlib.patches import FancyArrowPatch, Circle

# palette (harmonized with MUTED_DIV / MUTED_SEQ used in fig1/fig2)
INPUT_FC = "#DCE7F0"      # muted slate
FIRE_FC  = "#EAD08A"      # soft gold (the one mechanism that stands out)
SIG_FC   = "#F3EFE6"      # cream (= MUTED_DIV centre)
ACT_GRN  = "#2E8B57"      # muted sea green
ACT_GREY = "#D9D9D9"
ACT_TERRA= "#B5544A"      # terracotta (= MUTED_DIV high end)
EDGE     = "#9AA0A6"      # box edge / badge
ARR      = "#5F6368"      # arrows
HDR      = "#8A8A8A"      # column headers
INK      = "#222222"      # dark text

fig, ax = plt.subplots(figsize=(FULL_W, 3.05))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

BOX = {}


def box(name, x, y, w, h, text, fc, tc=INK, fs=FONT_SIZE - 1, weight="normal"):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=1.6",
                 fc=fc, ec=EDGE, lw=0.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight=weight, zorder=3)
    BOX[name] = (x, y, w, h)


def edge(name, side):
    x, y, w, h = BOX[name]
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2),
            "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


def arrow(p1, p2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10,
                 color=ARR, lw=1.3, shrinkA=1, shrinkB=3,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def header(x, text):
    ax.text(x, 60.3, text, ha="center", va="center", fontsize=FONT_SIZE - 2,
            color=HDR, weight="bold")


# column headers
header(11.5, "INPUT"); header(36.5, "MODEL ENTRY")
header(63.5, "LABEL-FREE SIGNAL"); header(89, "DEPLOYMENT ACTION")

# col 1 — inputs
box("ride", 2, 39.5, 19, 13, "New-context ride\non the reference\ncorridor (unlabeled)", INPUT_FC)
box("det",  2, 9.5, 19, 13, "Frozen detector\n(VQTransAE),\nweights fixed", INPUT_FC)

# col 2 — model-entry firewall
box("fw", 29, 24.5, 15, 14, "Per-ride\n$z$-score\nfirewall", FIRE_FC, fs=FONT_SIZE - 1)

# col 3 — three signals
box("s1", 52, 40.5, 23, 11, "Two-sided\nbudget check", SIG_FC)
box("s2", 52, 26.0, 23, 11, "$\\rho(s_t,\\mathrm{VDV})$\nhealth check", SIG_FC)
box("s3", 52, 11.5, 23, 11, "Codebook-usage\nJS divergence", SIG_FC)

# col 4 — actions
box("a1", 80, 40.5, 18, 11, "Recalibrate\nthreshold\n(conformal)", ACT_GRN, tc="white", fs=FONT_SIZE - 2)
box("a2", 80, 26.0, 18, 11, "Audit ranking /\ntriage fault", ACT_GREY, fs=FONT_SIZE - 2)
box("a3", 80, 11.5, 18, 11, "Flag stack fault;\ncodebook re-anchor", ACT_TERRA, tc="white", fs=FONT_SIZE - 2)

# arrows: inputs converge into the firewall
arrow(edge("ride", "r"), (29, 34), rad=-0.12)
arrow(edge("det", "r"), (29, 29), rad=0.12)
# firewall fans out to the three signals
arrow((44, 33), edge("s1", "l"), rad=-0.12)
arrow(edge("fw", "r"), edge("s2", "l"))
arrow((44, 30), edge("s3", "l"), rad=0.12)
# each signal drives its action (aligned rows)
for s, a in (("s1", "a1"), ("s2", "a2"), ("s3", "a3")):
    arrow(edge(s, "r"), edge(a, "l"))

# numbered badges on the three lanes (sit on each signal's left edge)
for k, s in enumerate(("s1", "s2", "s3"), 1):
    _, y, _, h = BOX[s][1], BOX[s][1] + BOX[s][3] / 2, 0, 0
    cy = BOX[s][1] + BOX[s][3] / 2
    ax.add_patch(Circle((52, cy), 1.7, fc=EDGE, ec="white", lw=0.8, zorder=4))
    ax.text(52, cy, str(k), ha="center", va="center", color="white",
            fontsize=FONT_SIZE - 3, weight="bold", zorder=5)

ax.text(50, 2.0, "one unlabeled corridor ride $\\rightarrow$ audit $+$ recalibrate $+$ diagnose, "
        "no labels, no retraining", ha="center", fontsize=FONT_SIZE - 2, style="italic", color="#666666")
save_fig(fig, "fig0_protocol")
