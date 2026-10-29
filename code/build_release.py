#!/usr/bin/env python3
"""
Build the de-identified data release for "When Not to Recalibrate".

GPS handling ("marked, not published"): latitude/longitude, absolute timestamps,
device identifiers, heart rate, sound, and other auxiliary channels are dropped.
Each row keeps
  gps_fix           1 if the row had a valid position fix, else 0
  along_track_m     cumulative distance travelled since the first fix (m)
and, on the analysed corridors,
  chainage_m        distance along the OSM corridor axis (N St / East Bay St)
  offset_m          signed lateral offset from the mapped line (m)
  in_corridor       1 if the row is inside the corridor used in the paper
Distances are relative to OSM nodes listed in corridors.csv, not coordinates.
"""
import json, math, re, shutil, pathlib, glob
import numpy as np, pandas as pd
import aic2_exp as X
import aic2_chainage as C
import aic2_city_osm as O
from aic2_city import IP

OUT = pathlib.Path("/home/kewei/paper_polish/release/when-not-to-recalibrate")
CAMP = {"autumn": "2023-10", "winter": "2023-11_12", "spring": "2024-01", "summer": "2024-05"}
ROUTES = ["11th", "13th", "16th", "Nst", "billy"]
COORD_COLS = re.compile(r"(^|_)(lat|lon|lng|latitude|longitude|prj_lat|prj_lng)($|_)", re.I)


def haversine_track(lat, lon):
    ok = ~(np.isnan(lat) | np.isnan(lon)) & (lat != 0) & (lon != 0)
    d = np.zeros(len(lat))
    idx = np.where(ok)[0]
    if len(idx) > 1:
        la, lo = np.radians(lat[idx]), np.radians(lon[idx])
        a = np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
        step = 2 * 6371000 * np.arcsin(np.sqrt(a))
        d[idx[1:]] = np.cumsum(step)
        d = np.maximum.accumulate(d)            # carry forward across rows without a fix
    return ok.astype(int), d


def rel_time(df):
    if "Date" not in df.columns:
        return np.arange(len(df)) / X.FS
    t = pd.to_datetime(df["Date"], errors="coerce")
    if t.isna().mean() > 0.01:                  # Excel-mangled column (spring/billy)
        return np.arange(len(df)) / X.FS
    return (t - t.iloc[0]).dt.total_seconds().values


def nst_cols(lat, lon, ch, e1):
    s, y = ch(C.enu(lat, lon))
    bad = np.isnan(lat) | np.isnan(lon)
    s[bad] = np.nan; y[bad] = np.nan
    return s, y, ((s >= 0) & (s <= e1)).astype(int)


def lincoln(ch, e1, man):
    for camp, tag in CAMP.items():
        for r in ROUTES:
            p = X.IPHONE / camp / f"{r}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p, low_memory=False)
            lat = pd.to_numeric(df["Latitude"], errors="coerce").values
            lon = pd.to_numeric(df["Longitude"], errors="coerce").values
            out = pd.DataFrame({"t_s": rel_time(df)})
            for c in ("X", "Y", "Z"):
                out[c] = pd.to_numeric(df[c], errors="coerce").values
            out["gps_fix"], out["along_track_m"] = haversine_track(lat, lon)
            if r == "Nst":
                out["chainage_m"], out["offset_m"], out["in_corridor"] = nst_cols(lat, lon, ch, e1)
            f = OUT / "data" / "lincoln_iphone" / tag / f"{r}.csv"
            f.parent.mkdir(parents=True, exist_ok=True)
            out.round(6).to_csv(f, index=False); man.append((f, len(out)))
    for r in ["11th", "16th", "Nst", "billy"]:
        p = X.CODE.parent / "infer" / "device_comparison" / f"android_{r}.csv"
        df = pd.read_csv(p, low_memory=False)
        lat, lon = df["prj_lat"].values.astype(float), df["prj_lng"].values.astype(float)
        out = pd.DataFrame({"t_s": np.arange(len(df)) / X.FS})
        for c in ("X", "Y", "Z"):
            out[c] = df[c].values
        out["gps_fix"], out["along_track_m"] = haversine_track(lat, lon)
        out["route_dist_m"] = df["route_dis"].values      # upstream map-matched distance
        if r == "Nst":
            out["chainage_m"], out["offset_m"], out["in_corridor"] = nst_cols(lat, lon, ch, e1)
        f = OUT / "data" / "lincoln_android" / f"{r}.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        out.round(6).to_csv(f, index=False); man.append((f, len(out)))


def charleston(man):
    nodes, ways = O.load_osm()
    line = O.street_centerline(nodes, ways)
    xs = O.intersections(nodes, ways)
    xs_pts = O.xy(*np.array([v[0] for v in xs.values()]).T)
    xs_d, xs_s, _ = O.project(xs_pts, line)
    keep = xs_d < 25
    xs_s_all = xs_s[keep]; xs_ids = [k for k, kk in zip(xs.keys(), keep) if kk]
    corr = {}
    for a, b in [(6, 16), (3, 12)]:
        fA = pd.read_csv(IP[a - 1], low_memory=False); fB = pd.read_csv(IP[b - 1], low_memory=False)
        mA, mB, ends, (sA, oA), (sB, oB) = O.osm_masks(fA, fB, line, xs_s_all, 15, 0, xs_ids, (2, 98))
        n0, n1 = O.osm_masks.last_nodes
        corr[a] = (sA, oA, mA); corr[b] = (sB, oB, mB)
        corr[(a, b)] = (n0, n1, round(ends[1] - ends[0]))
    for i, p in enumerate(IP, start=1):
        df = pd.read_csv(p, low_memory=False)
        lat = pd.to_numeric(df["Latitude"], errors="coerce").values
        lon = pd.to_numeric(df["Longitude"], errors="coerce").values
        out = pd.DataFrame({"t_s": np.arange(len(df)) / X.FS})
        for c in ("X", "Y", "Z"):
            out[c] = pd.to_numeric(df[c], errors="coerce").values
        out["gps_fix"], out["along_track_m"] = haversine_track(lat, lon)
        if i in corr:
            s, o, m = corr[i]
            out["chainage_m"], out["offset_m"], out["in_corridor"] = s, o, m.astype(int)
        f = OUT / "data" / "charleston_citadel_iphone" / f"segment_{i:02d}.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        out.round(6).to_csv(f, index=False); man.append((f, len(out)))
    rav = X.CODE / "code_version_sep18" / "New_Android_Ravenel_Bridge"
    for name in ("TotalAcceleration", "Gravity"):
        df = pd.read_csv(rav / f"{name}.csv")[["seconds_elapsed", "x", "y", "z"]]
        f = OUT / "data" / "charleston_ravenel_android" / f"{name}.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, index=False); man.append((f, len(df)))
    return corr


def citadel_android(man):
    """Citadel Android segments (gravity-removed m/s^2, ~1 Hz GPS, no timestamps) and the
    street-matched pair corridors of the extended decision check, as row intervals."""
    import aic2_citadel_android as CA
    P = CA.passes()
    for j, path in enumerate(CA.AND, start=1):
        df = pd.read_csv(path)
        lat, lon = df.location_latitude.values, df.location_longitude.values
        out = pd.DataFrame({"t_s": np.arange(len(df)) / X.FS})
        for c, k in zip(("X", "Y", "Z"), ("accelerometer_x", "accelerometer_y", "accelerometer_z")):
            out[c] = df[k].values
        out["gps_fix"], out["along_track_m"] = haversine_track(lat, lon)
        f = OUT / "data" / "charleston_citadel_android" / f"segment_{j:02d}.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        out.round(6).to_csv(f, index=False); man.append((f, len(out)))
    pairs, masks = CA.pairs(P, save=False)
    rows = []
    for (street, a, b), mk in masks.items():
        for k, m in zip((a, b), mk):
            idx = np.where(np.diff(np.r_[0, m.astype(int), 0]))[0]
            for r0, r1 in zip(idx[::2], idx[1::2]):
                rows.append(dict(street=street, pass_a=a, pass_b=b, pass_=k, row_from=int(r0), row_to=int(r1)))
    f = OUT / "data" / "charleston_citadel_pair_corridors.csv"
    pd.DataFrame(rows).rename(columns={"pass_": "pass"}).to_csv(f, index=False); man.append((f, len(rows)))


def columbia(man):
    import aic2_columbia as CO
    nodes, ways = CO.load_osm()
    R = {k: CO.load(k)[1] for k in CO.RIDES}
    extra = {}
    for tag, name, ends in [("main", "Main Street", ("113980212", "2695777079")),
                            ("sumter", "Sumter Street", ("4802607678", "114023465"))]:
        ch, line = CO.street_line(nodes, ways, name, *ends)
        ids, xs_s, nm = CO.xs_on(nodes, ways, name, ch, line)
        mA, mB, info, _, _ = CO.masks(R["R2a"], R["R2b"], line, xs_s, ids)
        for k, mk in (("R2a", mA), ("R2b", mB)):
            _, sv, ov = O.project(CO.xy(R[k].Latitude.values, R[k].Longitude.values), line)
            extra.setdefault(k, {}).update({f"{tag}_chainage_m": sv, f"{tag}_offset_m": ov, f"{tag}_in_corridor": mk.astype(int)})
        extra[tag] = (name, info)
    for k in CO.RIDES:
        df = R[k]
        lat = pd.to_numeric(df["Latitude"], errors="coerce").values
        lon = pd.to_numeric(df["Longitude"], errors="coerce").values
        out = pd.DataFrame({"t_s": rel_time(df)})
        for c in ("X", "Y", "Z"):
            out[c] = pd.to_numeric(df[c], errors="coerce").values
        out["gps_fix"], out["along_track_m"] = haversine_track(lat, lon)
        for c, v in extra.get(k, {}).items():
            out[c] = v
        f = OUT / "data" / "columbia_iphone" / f"{k}.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        out.round(6).to_csv(f, index=False); man.append((f, len(out)))
    return [dict(corridor=f"Columbia {extra[t][0]} (co-primary)", street=extra[t][0], from_node=extra[t][1][0],
                 to_node=extra[t][1][1], length_m=extra[t][1][2], used_by=f"columbia_iphone/R2a.csv, R2b.csv ({t}_* columns)")
            for t in ("main", "sumter")]


def corridors(corr, e1, more=()):
    rows = [dict(corridor="Lincoln N St (reference)", street="N Street",
                 from_node=C.N_S11, to_node=C.N_S21, length_m=round(e1),
                 used_by="lincoln_iphone/*/Nst.csv, lincoln_android/Nst.csv"),
            dict(corridor="Charleston segments 6/16", street="East Bay Street",
                 from_node=corr[(6, 16)][0], to_node=corr[(6, 16)][1], length_m=corr[(6, 16)][2],
                 used_by="segment_06.csv, segment_16.csv"),
            dict(corridor="Charleston segments 3/12", street="East Bay Street",
                 from_node=corr[(3, 12)][0], to_node=corr[(3, 12)][1], length_m=corr[(3, 12)][2],
                 used_by="segment_03.csv, segment_12.csv")]
    rows += list(more)
    pd.DataFrame(rows).to_csv(OUT / "corridors.csv", index=False)
    return rows


def results():
    dst = OUT / "results"; dst.mkdir(parents=True, exist_ok=True)
    dropped = {}
    for p in sorted(X.OUT.glob("*.csv")):
        df = pd.read_csv(p)
        bad = [c for c in df.columns if COORD_COLS.search(str(c))]
        dropped[p.name] = bad
        df.drop(columns=bad).to_csv(dst / p.name, index=False)
    for p in sorted(X.OUT.glob("*.json")):
        shutil.copy(p, dst / p.name)
    return dropped


def code():
    dst = OUT / "code"; dst.mkdir(parents=True, exist_ok=True)
    here = pathlib.Path(__file__).resolve().parent
    for p in sorted(here.glob("*.py")):
        shutil.copy(p, dst / p.name)
    shutil.copy(X.OUT / "protocol_ablation.py", dst / "protocol_ablation.py")
    for p in sorted((X.OUT / "figs").glob("gen_fig*.py")) + [X.OUT / "figs" / "paper_plot_style.py"]:
        (dst / "figures").mkdir(exist_ok=True); shutil.copy(p, dst / "figures" / p.name)
    shutil.copy(X.CODE / "aic_hi_validation_win65.py", dst / "aic_hi_validation_win65.py")
    ck = OUT / "checkpoint"; ck.mkdir(exist_ok=True)
    shutil.copy(X.CKPT, ck / "vqtransae_win65.pth")


def audit():
    """No file under data/ or results/ may carry a coordinate-like column or value."""
    hits = []
    for p in list((OUT / "data").rglob("*.csv")) + list((OUT / "results").glob("*.csv")):
        cols = pd.read_csv(p, nrows=0).columns
        hits += [(str(p), c) for c in cols if COORD_COLS.search(str(c))]
        txt = p.read_text()[:200000]
        if re.search(r"\b(40\.8[0-9]{3,}|32\.7[0-9]{3,}|33\.9[0-9]{3,}|-96\.7[0-9]{3,}|-79\.9[0-9]{3,}|-81\.0[0-9]{3,})", txt):
            hits.append((str(p), "coordinate-like value"))
    return hits


if __name__ == "__main__":
    for sub in ("data", "results", "code", "checkpoint"):          # keep .git, README, LICENSE
        if (OUT / sub).exists():
            shutil.rmtree(OUT / sub)
    (OUT / "data").mkdir(parents=True)
    ch, e0, e1 = C.axis()
    man = []
    lincoln(ch, e1, man)
    corr = charleston(man)
    col = columbia(man)
    citadel_android(man)
    rows = corridors(corr, e1, col)
    dropped = results()
    code()
    pd.DataFrame([(str(f.relative_to(OUT)), n) for f, n in man], columns=["file", "rows"]).to_csv(
        OUT / "MANIFEST.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("coordinate columns dropped from results:", {k: v for k, v in dropped.items() if v})
    print("audit hits:", audit())
