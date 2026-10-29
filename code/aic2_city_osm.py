#!/usr/bin/env python3
"""
Charleston cross-pass test with road-network matching and fixed geographic
corridor endpoints, plus the retained/excluded-row map.

Both street pairs (segments 6/16 and 3/12) run along East Bay Street. The
street centerline comes from OpenStreetMap (osm/citadel_osm.xml, fetched from
api.openstreetmap.org). Rows are matched to that named street, and the
corridor for each pair runs between two OSM intersection nodes that BOTH
passes cover, so the endpoints are fixed by the map rather than inferred from
the trajectories. Smoothing support must be complete (all 11 windows inside
the mask, none truncated at the recording ends).

Rules compared, everything else held fixed:
  R0   rounded lat/lon cell intersection, 4 dp (the published rule)
  R0s  R0 under 100 random sub-cell translations of the grid (sensitivity)
  OSM  matched to East Bay St within 15 m, between fixed intersections
  OSMi as OSM with both corridor ends moved 25 m inward
  OSM8 as OSM with an 8 m radius (excludes the mapped sidewalk band)
"""
import math, pathlib, time, urllib.request, io
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import aic2_exp as X
import aic2_codebook as CB
from aic2_city import (IP, load_citadel_segment, window_mask_from_rows, dev,
                       trim_to_full_smoothing_support)

ALPHA = X.ALPHA
N_C = int(np.ceil(1 / ALPHA)) - 1
K = X.K_SMOOTH
OSMDIR = X.OUT / "osm"
LAT0, LON0 = 32.783, -79.928
ROADS = {"primary", "secondary", "tertiary", "residential", "unclassified",
         "primary_link", "secondary_link", "tertiary_link"}


def xy(lat, lon):
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    return np.c_[np.radians(lon - LON0) * 6371000 * math.cos(math.radians(LAT0)),
                 np.radians(lat - LAT0) * 6371000]


def inv_xy(P):
    lon = np.degrees(P[:, 0] / (6371000 * math.cos(math.radians(LAT0)))) + LON0
    lat = np.degrees(P[:, 1] / 6371000) + LAT0
    return lat, lon


# ----------------------------------------------------------------- OSM street
def load_osm():
    r = ET.parse(OSMDIR / "citadel_osm.xml").getroot()
    nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in r.iter("node")}
    ways = []
    for w in r.iter("way"):
        t = {x.get("k"): x.get("v") for x in w.iter("tag")}
        if "highway" in t:
            ways.append((t, [nd.get("ref") for nd in w.iter("nd")]))
    return nodes, ways


def street_centerline(nodes, ways, name="East Bay Street"):
    """The named street's OSM ways form one connected, unbranched polyline
    (checked: 124 nodes, no node of degree > 2, two ends). Walk it end to end
    and use the mapped geometry unmodified, ordered south to north."""
    import collections
    adj = collections.defaultdict(set)
    for t, refs in ways:
        if t.get("name") == name:
            for a, b in zip(refs, refs[1:]):
                adj[a].add(b); adj[b].add(a)
    assert max(len(v) for v in adj.values()) <= 2, "street polyline branches"
    ends = [k for k, v in adj.items() if len(v) == 1]
    assert len(ends) == 2, "street polyline is not a single chain"
    path, prev = [ends[0]], None
    while len(path) < len(adj):
        nxt = [x for x in adj[path[-1]] if x != prev]
        prev = path[-1]; path.append(nxt[0])
    P = xy(*np.array([nodes[x] for x in path]).T)
    return P if P[0, 1] < P[-1, 1] else P[::-1]


def intersections(nodes, ways, name="East Bay Street"):
    street = {x for t, refs in ways if t.get("name") == name for x in refs}
    out = {}
    for t, refs in ways:
        if t.get("name") == name or t.get("highway") not in ROADS:
            continue
        for x in set(refs) & street:
            out[x] = t.get("name", "(unnamed)")
    return {k: (nodes[k], v) for k, v in out.items()}  # keyed by OSM node id


def project(P, line):
    a, b = line[:-1], line[1:]; ab = b - a
    L = np.linalg.norm(ab, axis=1); cum = np.r_[0, np.cumsum(L)]
    best_d = np.full(len(P), np.inf); best_s = np.zeros(len(P)); sgn = np.zeros(len(P))
    for i in range(len(a)):
        t = np.clip(((P - a[i]) @ ab[i]) / max(L[i] ** 2, 1e-9), 0, 1)
        q = a[i] + t[:, None] * ab[i]; w = P - q
        d = np.linalg.norm(w, axis=1); upd = d < best_d
        best_d[upd] = d[upd]; best_s[upd] = cum[i] + t[upd] * L[i]
        sgn[upd] = np.sign(ab[i][0] * w[upd, 1] - ab[i][1] * w[upd, 0])
    return best_d, best_s, sgn * best_d


# ------------------------------------------------------------------- masks
def strict_support(mask):
    n = len(mask); out = np.zeros(n, bool)
    for i in np.where(mask)[0]:
        if i - K >= 0 and i + K < n and mask[i - K:i + K + 1].all():
            out[i] = True
    return out


def cell_masks(fA, fB, dec=4, shift=(0.0, 0.0)):
    q = 10.0 ** dec
    def cells(f):
        return list(zip(np.floor(f.Latitude.values * q + shift[0]).astype(np.int64),
                        np.floor(f.Longitude.values * q + shift[1]).astype(np.int64)))
    if shift == (0.0, 0.0):                         # the published rule: round()
        cA = list(zip(fA.Latitude.round(dec), fA.Longitude.round(dec)))
        cB = list(zip(fB.Latitude.round(dec), fB.Longitude.round(dec)))
    else:
        cA, cB = cells(fA), cells(fB)
    sh = set(cA) & set(cB)
    return np.array([c in sh for c in cA]), np.array([c in sh for c in cB])


def osm_masks(fA, fB, line, xs_s, radius=15.0, inset=0.0, xs_ids=None, cov=(2, 98)):
    PA = xy(fA.Latitude.values, fA.Longitude.values)
    PB = xy(fB.Latitude.values, fB.Longitude.values)
    dA, sA, oA = project(PA, line); dB, sB, oB = project(PB, line)
    onA, onB = dA <= radius, dB <= radius
    covA = np.percentile(sA[onA], cov); covB = np.percentile(sB[onB], cov)
    lo, hi = max(covA[0], covB[0]), min(covA[1], covB[1])
    order = np.argsort(xs_s)
    inside = [i for i in order if lo <= xs_s[i] <= hi]   # intersections both passes cover
    i0, i1 = inside[0], inside[-1]
    e0, e1 = xs_s[i0] + inset, xs_s[i1] - inset
    osm_masks.last_nodes = (xs_ids[i0], xs_ids[i1]) if xs_ids is not None else None
    mA = onA & (sA >= e0) & (sA <= e1); mB = onB & (sB >= e0) & (sB <= e1)
    return mA, mB, (e0, e1), (sA, oA), (sB, oB)


# ------------------------------------------------------------------- scoring
def score_direction(sa, wa, sb, wb, fr):
    ncal = int(wa.sum())
    frozen = float(np.mean(sb[wb] > fr["tau"]))
    if ncal >= N_C:
        tau = X.conformal_tau(sa[wa], ALPHA); conf = float(np.mean(sb[wb] > tau)); st = "finite"
    else:
        tau, conf, st = np.inf, 0.0, "insufficient"
    return dict(n_cal=ncal, n_eval=int(wb.sum()), exceed_frozen=int((sb[wb] > fr["tau"]).sum()),
                exceed_conf=int((sb[wb] > tau).sum()) if np.isfinite(tau) else 0,
                CER_frozen=round(frozen, 4), CER_conformal=round(conf, 4),
                dev_frozen=round(abs(frozen - ALPHA), 4), dev_conformal=round(abs(conf - ALPHA), 4),
                status=st)


# ------------------------------------------------------------------- tiles
def fetch_tiles(lat_min, lat_max, lon_min, lon_max, z,
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"):
    def tx(lon): return (lon + 180) / 360 * 2 ** z
    def ty(lat):
        r = math.radians(lat); return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * 2 ** z
    x0, x1 = int(tx(lon_min)), int(tx(lon_max)); y0, y1 = int(ty(lat_max)), int(ty(lat_min))
    cache = OSMDIR / "tiles"; cache.mkdir(exist_ok=True)
    W = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            f = cache / f"esri_{z}_{x}_{y}.jpg"
            if not f.exists():
                req = urllib.request.Request(url.format(z=z, x=x, y=y),
                                             headers={"User-Agent": "renkw-research/1.0"})
                f.write_bytes(urllib.request.urlopen(req, timeout=30).read()); time.sleep(0.05)
            W.paste(Image.open(f).convert("RGB"), ((x - x0) * 256, (y - y0) * 256))
    def lon_of(px): return px / 2 ** z * 360 - 180
    def lat_of(py): n = math.pi - 2 * math.pi * py / 2 ** z; return math.degrees(math.atan(math.sinh(n)))
    ext = (lon_of(x0), lon_of(x1 + 1), lat_of(y1 + 1), lat_of(y0))   # l, r, b, t
    return W, ext


def merc_y(lat):
    """Web Mercator y expressed in degrees, so that it shares units with
    longitude on the x axis and an equal aspect ratio is geographically true."""
    r = np.radians(lat); return np.degrees(np.log(np.tan(np.pi / 4 + r / 2)))


def main():
    nodes, ways = load_osm()
    line = street_centerline(nodes, ways)
    xs = intersections(nodes, ways)
    xs_pts = xy(*np.array([v[0] for v in xs.values()]).T)
    xs_d, xs_s, _ = project(xs_pts, line)
    keep = xs_d < 25
    xs_s_all = xs_s[keep]; xs_names = [v[1] for v, k in zip(xs.values(), keep) if k]
    xs_ids = [k_ for k_, k in zip(xs.keys(), keep) if k]
    xs_ll = [v[0] for v, k in zip(xs.values(), keep) if k]

    m = X.A.load_model(X.CKPT, dev)
    src, _ = X.load_xyz(str(X.IPHONE / "summer" / "Nst.csv"))
    n0 = CB.windows(src); E0, D0 = X.raw_ED(m, src, dev)
    cal0 = np.zeros(n0, bool); cal0[: n0 // 2] = True
    fr = X.calibrate(E0, D0, cal0)

    rows, maps = [], {}
    rng = np.random.default_rng(0)
    for segA, segB in [(6, 16), (3, 12)]:
        dA, fA = load_citadel_segment(IP[segA - 1]); dB, fB = load_citadel_segment(IP[segB - 1])
        fA = fA.assign(Latitude=pd.to_numeric(fA.Latitude), Longitude=pd.to_numeric(fA.Longitude))
        fB = fB.assign(Latitude=pd.to_numeric(fB.Latitude), Longitude=pd.to_numeric(fB.Longitude))
        EA, DA = X.raw_ED(m, dA, dev); EB, DB = X.raw_ED(m, dB, dev)
        smA, smB = X.apply_frozen(EA, DA, fr), X.apply_frozen(EB, DB, fr)
        nA, nB = CB.windows(dA), CB.windows(dB)

        def run(rule, mA, mB, extra=None, trim=None):
            trim = trim or strict_support
            wA = trim(window_mask_from_rows(mA, nA))
            wB = trim(window_mask_from_rows(mB, nB))
            for ta, sa, wa, tb, sb, wb in [(segA, smA, wA, segB, smB, wB), (segB, smB, wB, segA, smA, wA)]:
                rows.append(dict(rule=rule, direction=f"{ta}->{tb}", **(extra or {}),
                                 **score_direction(sa, wa, sb, wb, fr)))
            return wA, wB

        run("R0 published (shared rounded cells, end-truncated support)", *cell_masks(fA, fB),
            trim=trim_to_full_smoothing_support)
        run("R0 strict support", *cell_masks(fA, fB))
        # R0 grid-position sensitivity: n_cal range over random sub-cell translations
        ncals = {f"{segA}->{segB}": [], f"{segB}->{segA}": []}
        for _ in range(100):
            sh = tuple(rng.uniform(0, 1, 2))
            mA, mB = cell_masks(fA, fB, shift=sh)
            ncals[f"{segA}->{segB}"].append(int(strict_support(window_mask_from_rows(mA, nA)).sum()))
            ncals[f"{segB}->{segA}"].append(int(strict_support(window_mask_from_rows(mB, nB)).sum()))
        for d, v in ncals.items():
            rows.append(dict(rule="R0s grid-shift sensitivity", direction=d, n_cal=f"{min(v)}-{max(v)}",
                             status=f"{np.mean(np.array(v) < N_C) * 100:.0f}% of shifts insufficient"))
        for rule, rad, inset, cov in [("OSM East Bay St 15 m", 15, 0, (2, 98)),
                                      ("OSM inset 25 m", 15, 25, (2, 98)),
                                      ("OSM narrower radius 8 m", 8, 0, (2, 98)),
                                      ("OSM coverage 0-100 pct", 15, 0, (0, 100)),
                                      ("OSM coverage 5-95 pct", 15, 0, (5, 95))]:
            mA, mB, ends, pa, pb = osm_masks(fA, fB, line, xs_s_all, rad, inset, xs_ids, cov)
            n0_, n1_ = osm_masks.last_nodes
            wA, wB = run(rule, mA, mB, dict(corridor_m=round(ends[1] - ends[0]),
                                            end_nodes=f"{n0_}({xs_names[xs_ids.index(n0_)]})-"
                                                      f"{n1_}({xs_names[xs_ids.index(n1_)]})"))
            if rule == "OSM East Bay St 15 m":
                maps[(segA, segB)] = dict(fA=fA, fB=fB, mA=mA, mB=mB, ends=ends, pa=pa, pb=pb,
                                          names=(xs_names[xs_ids.index(n0_)], xs_names[xs_ids.index(n1_)]))

    df = pd.DataFrame(rows)
    df.to_csv(X.OUT / "CITY_osm_robustness.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.to_string(index=False))

    # ---------------------------------------------------------------- figure
    plot_maps(maps, line, sidewalks(*load_osm()))


def sidewalks(nodes, ways):
    """Mapped sidewalk geometry (footway=sidewalk) as lat/lon polylines."""
    out = []
    for t, refs in ways:
        if t.get("highway") == "footway" and t.get("footway") == "sidewalk":
            ll = np.array([nodes[x] for x in refs if x in nodes])
            if len(ll) > 1:
                out.append(ll)
    return out


def point_at(line, s):
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(line) - 2))
    t = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
    return line[i] + t * (line[i + 1] - line[i])


def plot_maps(maps, line, walks):
    lat_l, lon_l = inv_xy(line)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 7.4), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    cols = {"A": "#E69F00", "B": "#56B4E9"}
    for ax, (pair, d) in zip(axes[:2], maps.items()):
        allat = np.r_[d["fA"].Latitude, d["fB"].Latitude]; allon = np.r_[d["fA"].Longitude, d["fB"].Longitude]
        pad = 0.00035
        la0, la1 = allat.min() - pad, allat.max() + pad
        lo0, lo1 = allon.min() - pad * 2.2, allon.max() + pad * 2.2
        img, ext = fetch_tiles(la0, la1, lo0, lo1, 19)
        ax.imshow(img, extent=(ext[0], ext[1], merc_y(ext[2]), merc_y(ext[3])))
        for w in walks:
            ax.plot(w[:, 1], merc_y(w[:, 0]), color="#F0E442", lw=0.9, alpha=0.9)
        ax.plot([], [], color="#F0E442", lw=0.9, label="mapped sidewalk (OSM)")
        ax.plot(lon_l, merc_y(lat_l), color="white", lw=1.0, ls="--", label="East Bay St (OSM)")
        for tag, f, mk, seg in [("A", d["fA"], d["mA"], pair[0]), ("B", d["fB"], d["mB"], pair[1])]:
            u = f.assign(k=mk).drop_duplicates(["Latitude", "Longitude"])
            ax.scatter(u.Longitude[~u.k], merc_y(u.Latitude[~u.k]), s=8, facecolors="none",
                       edgecolors=cols[tag], lw=0.6)
            ax.scatter(u.Longitude[u.k], merc_y(u.Latitude[u.k]), s=7, color=cols[tag],
                       label=f"segment {seg}")
            # travel direction: one arrow beside the track, at its middle
            ll = u[["Latitude", "Longitude"]].values; m_ = len(ll) // 2
            dx = 0.00028 if tag == "A" else -0.00028
            ax.annotate("", xy=(ll[m_ + 6, 1] + dx, merc_y(ll[m_ + 6, 0])),
                        xytext=(ll[m_ - 6, 1] + dx, merc_y(ll[m_ - 6, 0])),
                        arrowprops=dict(arrowstyle="-|>", color=cols[tag], lw=2.0, mutation_scale=12))
        for e, nm in zip(d["ends"], d["names"]):
            la, lo = inv_xy(point_at(line, e)[None])
            ax.plot(lo, merc_y(la), marker="s", ms=8, mfc="#D55E00", mec="white", mew=1.2, ls="none")
            ax.annotate(nm, (lo[0], merc_y(la)[0]), xytext=(8, 0), textcoords="offset points",
                        fontsize=7.5, color="white", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, lw=0))
        ax.plot([], [], marker="s", ms=7, mfc="#D55E00", mec="white", ls="none",
                label="corridor end (OSM intersection)")
        ax.plot([], [], marker="o", ms=4, mfc="none", mec="0.4", ls="none", label="open: excluded rows")
        ax.set_xlim(lo0, lo1); ax.set_ylim(merc_y(la0), merc_y(la1)); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"({'a' if pair == (6, 16) else 'b'}) segments {pair[0]} and {pair[1]}",
                     fontsize=10, loc="left")
        ax.legend(loc="lower left", fontsize=6.3, framealpha=0.88)
    ax = axes[2]
    bins = np.arange(-8, 18.5, 0.5)
    for i, (pair, d) in enumerate(maps.items()):
        for tag, (s_, o), mk, seg in [("A", d["pa"], d["mA"], pair[0]), ("B", d["pb"], d["mB"], pair[1])]:
            o = np.clip(o[mk], bins[0], bins[-1])
            ax.hist(o, bins=bins, histtype="step", lw=1.6, color=cols[tag], ls="-" if i == 0 else ":",
                    label=f"segment {seg} (median {np.median(o):+.1f} m)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("signed lateral offset from the mapped East Bay St line (m)")
    ax.set_ylabel("retained GPS rows")
    ax.legend(fontsize=7, title="retained rows only", title_fontsize=7)
    ax.set_title("(c) lateral position of each pass", fontsize=10, loc="left")
    fig.text(0.01, 0.006, "Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community. "
             "Street and sidewalk data \u00a9 OpenStreetMap contributors, ODbL. "
             "Offsets beyond -8 / +18 m are drawn in the end bins.", fontsize=6.5, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = X.OUT / "figs" / "fig_city_spatial"
    fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png", dpi=200)
    print("figure ->", out)


if __name__ == "__main__":
    main()
