#!/usr/bin/env python3
"""Study-area map: (a) Lincoln routes, (b) Charleston Citadel tour + Ravenel ride, (c) Columbia rides.
GPS traces plotted on CARTO light tiles (OpenStreetMap data). Output: aic2_results/figs/fig_studymap.pdf/png."""
import math, time, urllib.request, glob, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import aic2_exp as X
from aic2_city import IP

CACHE = X.OUT / "osm" / "tiles"; CACHE.mkdir(parents=True, exist_ok=True)
URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
OI = dict(blue="#0072B2", verm="#D55E00", green="#009E73", orange="#E69F00", sky="#56B4E9",
          purple="#CC79A7", yellow="#F0E442", black="#000000")


def merc_y(lat):
    r = np.radians(lat); return np.degrees(np.log(np.tan(np.pi / 4 + r / 2)))


def tiles(la0, la1, lo0, lo1, z):
    tx = lambda lon: (lon + 180) / 360 * 2 ** z
    ty = lambda lat: (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * 2 ** z
    x0, x1 = int(tx(lo0)), int(tx(lo1)); y0, y1 = int(ty(la1)), int(ty(la0))
    W = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            f = CACHE / f"esrigray_{z}_{x}_{y}.jpg"
            if not f.exists():
                req = urllib.request.Request(URL.format(z=z, x=x, y=y), headers={"User-Agent": "renkw-research/1.0"})
                f.write_bytes(urllib.request.urlopen(req, timeout=30).read()); time.sleep(0.05)
            W.paste(Image.open(f).convert("RGB"), ((x - x0) * 256, (y - y0) * 256))
    lon_of = lambda px: px / 2 ** z * 360 - 180
    lat_of = lambda py: math.degrees(math.atan(math.sinh(math.pi - 2 * math.pi * py / 2 ** z)))
    return W, (lon_of(x0), lon_of(x1 + 1), lat_of(y1 + 1), lat_of(y0))


def ll(df, la="Latitude", lo="Longitude"):
    a = pd.to_numeric(df[la], errors="coerce").values; b = pd.to_numeric(df[lo], errors="coerce").values
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    return a[ok], b[ok]


def panel(ax, tracks, z, title, pad=0.002, scale_m=500):
    la = np.concatenate([t[0] for t in tracks]); lo = np.concatenate([t[1] for t in tracks])
    la0, la1, lo0, lo1 = la.min() - pad, la.max() + pad, lo.min() - pad * 1.3, lo.max() + pad * 1.3
    img, ext = tiles(la0, la1, lo0, lo1, z)
    ax.imshow(img, extent=(ext[0], ext[1], merc_y(ext[2]), merc_y(ext[3])))
    for a, b, col, lw, lab in tracks:
        ax.plot(b, merc_y(a), color=col, lw=lw, label=lab, solid_capstyle="round")
    ax.set_xlim(lo0, lo1); ax.set_ylim(merc_y(la0), merc_y(la1)); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10, loc="left")
    # scale bar
    dlon = scale_m / (111320 * math.cos(math.radians((la0 + la1) / 2)))
    x0 = lo0 + 0.06 * (lo1 - lo0); y0 = merc_y(la0) + 0.06 * (merc_y(la1) - merc_y(la0))
    ax.plot([x0, x0 + dlon], [y0, y0], color="black", lw=2.2)
    ax.text(x0 + dlon / 2, y0 + 0.012 * (merc_y(la1) - merc_y(la0)), f"{scale_m} m" if scale_m < 1000 else f"{scale_m//1000} km",
            ha="center", va="bottom", fontsize=7)
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15, 0.85]))
    # (a) Lincoln, May 2024 iPhone rides
    cols = {"11th": OI["sky"], "13th": OI["orange"], "16th": OI["green"], "Nst": OI["black"], "billy": OI["purple"]}
    names = {"11th": "11th St", "13th": "13th St", "16th": "16th St", "Nst": "N St (reference)", "billy": "Billy Wolff Trail"}
    tr = []
    for r in ["11th", "13th", "16th", "billy", "Nst"]:
        a, b = ll(pd.read_csv(X.IPHONE / "summer" / f"{r}.csv", low_memory=False))
        tr.append((a, b, cols[r], 2.6 if r == "Nst" else 1.6, names[r]))
    panel(axes[0], tr, 14, "(a) Lincoln, NE", scale_m=500)
    # (b) Charleston: Citadel segments + Ravenel
    cit = [ll(pd.read_csv(p, low_memory=False)) for p in IP]
    rav = pd.read_csv(X.CODE / "code_version_sep18" / "New_Android_Ravenel_Bridge" / "Location.csv")
    ra, rb = ll(rav, "latitude", "longitude")
    tr = [(a, b, OI["blue"], 1.4, "Citadel tour (20 segments)" if i == 0 else None) for i, (a, b) in enumerate(cit)]
    tr.append((ra, rb, OI["verm"], 1.6, "Ravenel Bridge ride"))
    panel(axes[1], tr, 13, "(b) Charleston, SC", scale_m=1000)
    # (c) Columbia
    raw = pathlib.Path("/home/kewei/paper_polish/raw_extra/uofsc_feb2024/UofSC Bike Data Feb 2024_iOS")
    tr = []
    for k, col, lab in [("R1", OI["purple"], "R1"), ("R2a", OI["orange"], "R2a"), ("R2b", OI["sky"], "R2b")]:
        a, b = ll(pd.read_csv(next(raw.glob(f"{k}Untitled*.csv"))))
        tr.append((a, b, col, 1.6, lab))
    panel(axes[2], tr, 15, "(c) Columbia, SC", scale_m=200)
    fig.text(0.01, 0.01, "Base map: Esri, HERE, Garmin, © OpenStreetMap contributors, and the GIS user community.", fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = X.OUT / "figs" / "fig_studymap"
    fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png", dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
