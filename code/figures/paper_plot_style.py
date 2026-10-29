# paper_plot_style.py — shared style for Paper-2 figures (CACAIE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT_SIZE = 8          # printed at final size (factor-1.0 includegraphics) => 8pt, safely >=6pt
DPI = 300
# Exact CACAIE/cas-dc widths (measured: \columnwidth=238.25pt, \textwidth=494.51pt).
# Generate at these so \includegraphics uses width=\columnwidth / \textwidth (NO rescaling,
# which was shrinking single-column fonts to ~5.3pt, below the 6pt floor).
COL_W = 3.30           # single-column figure width (in)
FULL_W = 6.84          # full-width (two-column) figure width (in)
from pathlib import Path
FIG_DIR = str(Path(__file__).resolve().parent)
RESULTS = str(Path(__file__).resolve().parents[1])

matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.usetex": False,
    "mathtext.fontset": "stix",
})

# Okabe–Ito colorblind-safe palette
OI = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "verm":    "#D55E00",
    "purple":  "#CC79A7",
    "grey":    "#999999",
}


from matplotlib.colors import LinearSegmentedColormap as _LSC
# Muted, print-friendly diverging map: slate-blue -> warm cream -> terracotta.
# Used for budget-centered drift matrices (elegant vs the harsh saturated RdBu_r).
MUTED_DIV = _LSC.from_list("muted_div", [
    "#3B6B8F", "#7CA3BF", "#C9DCE6", "#F3EFE6", "#E7B7A0", "#CC7B62", "#A8443A"])
# Muted sequential green (companion for ranking grids; softer than YlGn).
MUTED_SEQ = _LSC.from_list("muted_seq", [
    "#F3EFE6", "#DCE7C8", "#AECF8F", "#6FB05C", "#2E8B57", "#1B6B45"])


def refine_colorbar(cb, ticks=None):
    """Slim, frameless colorbar with short ticks — matched across figures."""
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)


def cell_text_color(value, norm, cmap):
    """Black or white text per cell by luminance of the mapped colour (works for
    sequential AND diverging maps; no hand-tuned value threshold)."""
    import numpy as _np
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    r, g, b, _ = cm(norm(value))
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "black"


def style_heatmap(ax, nrows, ncols):
    """Publication heatmap polish: square cells, thin white gridlines between cells,
    no tick marks, no spines."""
    import numpy as _np
    ax.set_aspect("equal")
    ax.set_xticks(_np.arange(-.5, ncols, 1), minor=True)
    ax.set_yticks(_np.arange(-.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def save_fig(fig, name):
    for fmt in ("pdf", "png"):
        fig.savefig(f"{FIG_DIR}/{name}.{fmt}")
    print(f"Saved: {FIG_DIR}/{name}.pdf + .png")
