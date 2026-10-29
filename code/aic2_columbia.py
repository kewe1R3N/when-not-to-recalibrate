#!/usr/bin/env python3
"""
Columbia, SC repeated pass (refine-logs/PREREG_columbia.md).
  --geometry : which named OSM streets both R2a and R2b follow (no scores)
  --run      : corridor, cross-pass CER, diagnostics, sensitivity
"""
import argparse, collections, json, math, pathlib, glob
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd
import aic2_exp as X
import aic2_codebook as CB
import aic2_city_osm as O
from aic2_city import window_mask_from_rows, dev

RAW = pathlib.Path("/home/kewei/paper_polish/raw_extra/uofsc_feb2024/UofSC Bike Data Feb 2024_iOS")
RIDES = {k: next(RAW.glob(f"{k}Untitled*.csv")) for k in ("R1", "R2a", "R2b")}
OSMF = X.OUT / "osm" / "columbia_osm.xml"
LAT0, LON0 = 33.990, -81.026


def xy(lat, lon):
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    return np.c_[np.radians(lon - LON0) * 6371000 * math.cos(math.radians(LAT0)),
                 np.radians(lat - LAT0) * 6371000]


def load_osm():
    r = ET.parse(OSMF).getroot()
    nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in r.iter("node")}
    ways = []
    for w in r.iter("way"):
        t = {x.get("k"): x.get("v") for x in w.iter("tag")}
        if "highway" in t:
            ways.append((t, [nd.get("ref") for nd in w.iter("nd") if nd.get("ref") in nodes]))
    return nodes, ways


def chains(nodes, ways, name):
    """Connected components of the named street, each split into unbranched chains."""
    adj = collections.defaultdict(set)
    for t, refs in ways:
        if t.get("name") == name:
            for a, b in zip(refs, refs[1:]):
                adj[a].add(b); adj[b].add(a)
    out, seen_edges = [], set()
    branch = {k for k, v in adj.items() if len(v) != 2}
    starts = sorted(branch) or [min(adj)]
    for s in starts:
        for nb in sorted(adj[s]):
            if (s, nb) in seen_edges:
                continue
            path = [s, nb]; seen_edges |= {(s, nb), (nb, s)}
            while path[-1] not in branch and path[-1] != s:
                nxt = [x for x in adj[path[-1]] if x != path[-2]]
                if not nxt: break
                seen_edges |= {(path[-1], nxt[0]), (nxt[0], path[-1])}; path.append(nxt[0])
            out.append(path)
    return out


def load(k):
    df = pd.read_csv(RIDES[k], low_memory=False)
    data = df[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    return data, df


def geometry():
    nodes, ways = load_osm()
    names = sorted({t["name"] for t, _ in ways if "name" in t})
    P = {k: xy(load(k)[1].Latitude.values, load(k)[1].Longitude.values) for k in ("R2a", "R2b")}
    rows = []
    for nm in names:
        for ch in chains(nodes, ways, nm):
            line = xy(*np.array([nodes[x] for x in ch]).T)
            if len(line) < 2: continue
            L = float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))
            r = dict(street=nm, chain_nodes=f"{ch[0]}-{ch[-1]}", chain_m=round(L))
            for k in P:
                d, s, _ = O.project(P[k], line)
                on = d <= 15
                r[f"{k}_rows"] = int(on.sum())
                r[f"{k}_span_m"] = round(float(np.ptp(s[on]))) if on.sum() > 20 else 0
            rows.append(r)
    df = pd.DataFrame(rows)
    df = df[(df.R2a_rows > 100) & (df.R2b_rows > 100)].sort_values("R2a_span_m", ascending=False)
    print(df.to_string(index=False))
    df.to_csv(X.OUT / "COL_geometry.csv", index=False)


def street_line(nodes, ways, name, n0, n1):
    for ch in chains(nodes, ways, name):
        if {ch[0], ch[-1]} == {n0, n1}:
            return ch, xy(*np.array([nodes[x] for x in ch]).T)
    raise KeyError(name)


def xs_on(nodes, ways, name, chain, line):
    st = set(chain); out = {}
    for t, refs in ways:
        if t.get("name") == name or t.get("highway") not in O.ROADS:
            continue
        for x in set(refs) & st:
            out[x] = t.get("name", "(unnamed)")
    ids = list(out); P = xy(*np.array([nodes[i] for i in ids]).T)
    d, s, _ = O.project(P, line)
    return [i for i, dd in zip(ids, d) if dd < 25], s[d < 25], [out[i] for i, dd in zip(ids, d) if dd < 25]


def masks(fA, fB, line, xs_s, xs_ids, radius=15, inset=0, cov=(2, 98)):
    PA = xy(fA.Latitude.values, fA.Longitude.values); PB = xy(fB.Latitude.values, fB.Longitude.values)
    dA, sA, oA = O.project(PA, line); dB, sB, oB = O.project(PB, line)
    onA, onB = dA <= radius, dB <= radius
    cA = np.percentile(sA[onA], cov); cB = np.percentile(sB[onB], cov)
    lo, hi = max(cA[0], cB[0]), min(cA[1], cB[1])
    order = np.argsort(xs_s); inside = [i for i in order if lo <= xs_s[i] <= hi]
    i0, i1 = inside[0], inside[-1]
    e0, e1 = xs_s[i0] + inset, xs_s[i1] - inset
    mA = onA & (sA >= e0) & (sA <= e1); mB = onB & (sB >= e0) & (sB <= e1)
    dirA = np.sign(np.nanmedian(np.diff(sA[mA]))); dirB = np.sign(np.nanmedian(np.diff(sB[mB])))
    return mA, mB, (xs_ids[i0], xs_ids[i1], round(e1 - e0)), (dirA, dirB), (oA[mA], oB[mB])


def run():
    from aic2_city import seg_rho
    from aic2_fault_injection import code_indices, usage_hist, js_div
    nodes, ways = load_osm()
    geo = pd.read_csv(X.OUT / "COL_geometry.csv")
    m = X.A.load_model(X.CKPT, dev)
    src, _ = X.load_xyz(str(X.IPHONE / "summer" / "Nst.csv"))
    n0 = CB.windows(src); E0, D0 = X.raw_ED(m, src, dev)
    cal0 = np.zeros(n0, bool); cal0[: n0 // 2] = True
    fr = X.calibrate(E0, D0, cal0)
    p_src = usage_hist(code_indices(m, src[: len(src) // 2]), 1024)
    R = {k: load(k) for k in RIDES}
    sm = {}
    for k in ("R2a", "R2b"):
        E, D = X.raw_ED(m, R[k][0], dev); sm[k] = X.apply_frozen(E, D, fr)
    nw = {k: CB.windows(R[k][0]) for k in ("R2a", "R2b")}
    lines = {}
    for g in geo.itertuples():
        n_a, n_b = g.chain_nodes.split("-")
        ch, line = street_line(nodes, ways, g.street, n_a, n_b)
        ids, xs_s, nm = xs_on(nodes, ways, g.street, ch, line)
        try:
            L = masks(R["R2a"][1], R["R2b"][1], line, xs_s, ids)[2][2]
        except (IndexError, ValueError):
            L = 0
        lines[(g.street, g.chain_nodes)] = (line, ids, xs_s, nm, L)
    order = sorted(lines, key=lambda k: -lines[k][4])
    print("corridor lengths:", [(k[0], lines[k][4]) for k in order])
    rows = []
    for rank, key in enumerate(order):
        line, ids, xs_s, nm, L0 = lines[key]
        if L0 == 0:
            continue
        top = lines[order[0]][4]
        role = "primary" if L0 == top else "secondary"
        variants = [("primary 15 m", 15, 0, (2, 98))]
        if role == "primary":
            variants += [("inset 25 m", 15, 25, (2, 98)), ("radius 8 m", 8, 0, (2, 98)),
                         ("coverage 0-100", 15, 0, (0, 100)), ("coverage 5-95", 15, 0, (5, 95))]
        for vname, rad, ins, cov in variants:
            try:
                mA, mB, (na, nb, L), (dA, dB), (oA, oB) = masks(R["R2a"][1], R["R2b"][1], line, xs_s, ids, rad, ins, cov)
            except (IndexError, ValueError) as e:
                rows.append(dict(corridor=key[0], role=role, variant=vname, status=f"no corridor ({type(e).__name__})")); continue
            wA = O.strict_support(window_mask_from_rows(mA, nw["R2a"]))
            wB = O.strict_support(window_mask_from_rows(mB, nw["R2b"]))
            for ta, wa, tb, wb in [("R2a", wA, "R2b", wB), ("R2b", wB, "R2a", wA)]:
                r = O.score_direction(sm[ta], wa, sm[tb], wb, fr)
                if r["status"] == "insufficient": out = "defer"
                elif r["n_eval"] == 0: out = "no evaluation windows"
                elif r["dev_conformal"] < r["dev_frozen"]: out = "passes"
                elif r["dev_conformal"] == r["dev_frozen"]: out = "tie; keep"
                else: out = "refuse"
                rows.append(dict(corridor=key[0], role=role, variant=vname, direction=f"{ta}->{tb}", corridor_m=L,
                                 end_nodes=f"{na}({nm[ids.index(na)]})-{nb}({nm[ids.index(nb)]})",
                                 travel_dir_same=bool(dA == dB),
                                 med_offset_a=round(float(np.median(oA)), 1) if len(oA) else None,
                                 med_offset_b=round(float(np.median(oB)), 1) if len(oB) else None,
                                 rows_a=int(mA.sum()), rows_b=int(mB.sum()), **r, audit=out))
    df = pd.DataFrame(rows); df.to_csv(X.OUT / "COL_crosspass.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(df.drop(columns=["end_nodes"], errors="ignore").to_string(index=False))
        print(df[df.variant == "primary 15 m"][["corridor", "direction", "end_nodes"]].to_string(index=False))
    diag = []
    for k in RIDES:
        rho, n = seg_rho(m, R[k][0])
        js = js_div(usage_hist(code_indices(m, R[k][0]), 1024), p_src)
        diag.append(dict(ride=k, rho_aux=round(float(rho), 3), n_windows=n, JS=round(float(js), 4)))
    print(pd.DataFrame(diag).to_string(index=False))
    pd.DataFrame(diag).to_csv(X.OUT / "COL_diagnostics.csv", index=False)


def figure():
    """(a) the three rides and the two co-primary corridors on imagery;
    (b) deployed score along Main St for R2a and R2b with the frozen threshold."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nodes, ways = load_osm()
    R = {k: load(k) for k in RIDES}
    cor = {}
    for name, ends in [("Sumter Street", ("4802607678", "114023465")), ("Main Street", ("113980212", "2695777079"))]:
        ch, line = street_line(nodes, ways, name, *ends)
        ids, xs_s, nm = xs_on(nodes, ways, name, ch, line)
        mA, mB, info, _, _ = masks(R["R2a"][1], R["R2b"][1], line, xs_s, ids)
        cor[name] = (line, mA, mB, info)
    allat = np.r_[R["R2a"][1].Latitude, R["R2b"][1].Latitude]; allon = np.r_[R["R2a"][1].Longitude, R["R2b"][1].Longitude]
    pad = 0.0006
    la0, la1, lo0, lo1 = allat.min() - pad, allat.max() + pad, allon.min() - pad, allon.max() + pad
    img, ext = O.fetch_tiles(la0, la1, lo0, lo1, 18)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.0), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    ax.imshow(img, extent=(ext[0], ext[1], O.merc_y(ext[2]), O.merc_y(ext[3])))
    cols = {"R2a": "#E69F00", "R2b": "#56B4E9"}
    for k in ("R2a", "R2b"):
        f = R[k][1]
        ax.plot(f.Longitude, O.merc_y(f.Latitude), color=cols[k], lw=1.0, alpha=0.9, label=f"ride {k}")
    for name, (line, mA, mB, info) in cor.items():
        f = R["R2a"][1][mA]
        ax.scatter(f.Longitude, O.merc_y(f.Latitude), s=4, color="white", zorder=3)
        mid = f.iloc[len(f) // 2]
        ax.annotate(name.replace(" Street", " St"), (mid.Longitude, O.merc_y(mid.Latitude)), xytext=(10, 0),
                    textcoords="offset points", fontsize=8, color="white", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0))
    ax.scatter([], [], s=10, color="white", label="co-primary corridors (Sumter St, Main St)")
    ax.set_xlim(lo0, lo1); ax.set_ylim(O.merc_y(la0), O.merc_y(la1)); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="lower left", fontsize=7.5, framealpha=0.88)
    ax.set_title("(a) Columbia rides", fontsize=10, loc="left")
    ax = axes[1]
    m = X.A.load_model(X.CKPT, dev)
    src, _ = X.load_xyz(str(X.IPHONE / "summer" / "Nst.csv"))
    n0 = CB.windows(src); E0, D0 = X.raw_ED(m, src, dev)
    cal0 = np.zeros(n0, bool); cal0[: n0 // 2] = True
    fr = X.calibrate(E0, D0, cal0)
    line, mA, mB, info = cor["Main Street"]
    for k, mk in (("R2a", mA), ("R2b", mB)):
        d, f = R[k]; n = CB.windows(d); E, D = X.raw_ED(m, d, dev); sm = X.apply_frozen(E, D, fr)
        w = O.strict_support(window_mask_from_rows(mk, n))
        _, s, _ = O.project(xy(f.Latitude.values, f.Longitude.values), line)
        sw = np.array([s[i * X.STEP:i * X.STEP + X.WIN].mean() for i in range(n)])
        ex = 100 * np.mean(sm[w] > fr["tau"])
        ax.plot(sw[w], sm[w], ".", ms=3, color=cols[k], label=f"ride {k}: {w.sum()} windows, {ex:.1f}% above")
    ax.axhline(fr["tau"], color="#D55E00", ls="--", lw=1.2, label="frozen Lincoln threshold")
    ax.set_xlabel("Chainage along Main St (m)"); ax.set_ylabel(r"Deployed score $\tilde s_t$")
    ax.set_ylim(0.1, 1.18); ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=1)
    ax.set_title("(b) Main St corridor", fontsize=10, loc="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.text(0.01, 0.01, "Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community; "
             "street data © OpenStreetMap contributors, ODbL.", fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext_ in ("pdf", "png"):
        fig.savefig(X.OUT / "figs" / f"fig_columbia.{ext_}", dpi=200)
    print("saved fig_columbia")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--geometry", action="store_true"); ap.add_argument("--run", action="store_true"); ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    if a.geometry: geometry()
    if a.run: run()
    if a.figure: figure()

