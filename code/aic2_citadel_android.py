#!/usr/bin/env python3
"""Citadel Android passes added to the out-of-sample decision check
(refine-logs/PREREG_citadel_android.md, revision 1).
  --gate0    : timebase and gravity convention (no scores)
  --pairs    : street-matched pairs and S/R categories (geometry only)
  --run      : scores, prompts, decisions
"""
import argparse, collections, math
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd, torch
import aic2_exp as X
import aic2_codebook as CB
import aic2_city_osm as O
from aic2_city import IP, load_citadel_segment, window_mask_from_rows, seg_rho
from aic2_decision_split import decide, split
import aic2_decision_split as DS

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AND = sorted((X.CODE / "data_process_Citadel_android").glob("filtered*.csv"),
             key=lambda p: int("".join(filter(str.isdigit, p.name))))
OSMF = X.OUT / "osm" / "citadel_full_osm.xml"
RAD, MIN_FOLLOW, N_C, MIN_HALF = 15.0, 200, 99, 50


def load_android(path):
    df = pd.read_csv(path)
    df = df.assign(Latitude=df.location_latitude.ffill().bfill(), Longitude=df.location_longitude.ffill().bfill())
    xyz = df[["accelerometer_x", "accelerometer_y", "accelerometer_z"]].fillna(0).values.astype(np.float32)
    return xyz, df


def load_iphone(i):
    d, f = load_citadel_segment(IP[i - 1])
    la = pd.to_numeric(f.Latitude, errors="coerce"); lo = pd.to_numeric(f.Longitude, errors="coerce")
    bad = (la == 0) | (lo == 0); la[bad] = np.nan; lo[bad] = np.nan
    return d, f.assign(Latitude=la, Longitude=lo)


def passes():
    P = {f"i{i}": load_iphone(i) for i in range(1, len(IP) + 1)}
    P.update({f"a{j}": load_android(p) for j, p in enumerate(AND, 1)})
    return P


def gate0():
    sp, nr = [], []
    for p in AND:
        d, f = load_android(p)
        g = f[["location_latitude", "location_longitude"]].dropna().drop_duplicates()
        P = O.xy(g.location_latitude.values, g.location_longitude.values)
        sp.append(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / (len(d) * 0.1))
        nr.append(np.median(np.linalg.norm(d, axis=1)))
    ms, mn = float(np.median(sp)), float(np.median(np.concatenate(
        [np.linalg.norm(load_android(p)[0], axis=1) for p in AND])))
    print(f"median speed {ms:.2f} m/s (pass 1.5-8); median |a| {mn:.3f} m/s^2")
    tb = 1.5 <= ms <= 8
    conv = "gravity-removed (linear)" if mn < 3 else "gravity-included" if abs(mn - 9.81) <= 1.5 else "STOP"
    print("timebase:", "pass" if tb else "STOP", "| convention:", conv)
    pd.DataFrame([dict(median_speed_ms=round(ms, 2), median_norm=round(mn, 3), timebase_pass=tb, convention=conv)]
                 ).to_csv(X.OUT / "CA_gate0.csv", index=False)


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
    adj = collections.defaultdict(set)
    for t, refs in ways:
        if t.get("name") == name:
            for a, b in zip(refs, refs[1:]):
                adj[a].add(b); adj[b].add(a)
    out, seen = [], set()
    branch = {k for k, v in adj.items() if len(v) != 2}
    for s in (sorted(branch) or [min(adj)]):
        for nb in sorted(adj[s]):
            if (s, nb) in seen:
                continue
            path = [s, nb]; seen |= {(s, nb), (nb, s)}
            while path[-1] not in branch and path[-1] != s:
                nxt = [x for x in adj[path[-1]] if x != path[-2]]
                if not nxt: break
                seen |= {(path[-1], nxt[0]), (nxt[0], path[-1])}; path.append(nxt[0])
            out.append(path)
    return out


def xs_on(nodes, ways, name, chain, line):
    st = set(chain); out = {}
    for t, refs in ways:
        if t.get("name") == name or t.get("highway") not in O.ROADS:
            continue
        for x in set(refs) & st:
            out[x] = t.get("name", "(unnamed)")
    if not out:
        return [], np.array([])
    ids = list(out); d, s, _ = O.project(O.xy(*np.array([nodes[i] for i in ids]).T), line)
    return [i for i, dd in zip(ids, d) if dd < 25], s[d < 25]


def corridor(fA, fB, line, xs_s):
    PA = O.xy(fA.Latitude.values, fA.Longitude.values); PB = O.xy(fB.Latitude.values, fB.Longitude.values)
    dA, sA, _ = O.project(PA, line); dB, sB, _ = O.project(PB, line)
    onA, onB = dA <= RAD, dB <= RAD
    cA = np.percentile(sA[onA], (2, 98)); cB = np.percentile(sB[onB], (2, 98))
    lo, hi = max(cA[0], cB[0]), min(cA[1], cB[1])
    inside = sorted(s for s in xs_s if lo <= s <= hi)
    if len(inside) < 2:
        return None
    e0, e1 = inside[0], inside[-1]
    mA = onA & (sA >= e0) & (sA <= e1); mB = onB & (sB >= e0) & (sB <= e1)
    # category S: median lateral distance between the two corridor tracks < 5 m and chainage overlap >= 80 %
    a, b = PA[mA], PB[mB]
    if len(a) == 0 or len(b) == 0:
        return None
    a_u, b_u = np.unique(a, axis=0), np.unique(b, axis=0)
    dab = np.median(np.min(np.linalg.norm(a_u[:, None] - b_u[None], axis=2), axis=1))
    dba = np.median(np.min(np.linalg.norm(b_u[:, None] - a_u[None], axis=2), axis=1))
    rA, rB = (sA[mA].min(), sA[mA].max()), (sB[mB].min(), sB[mB].max())
    ov = max(0, min(rA[1], rB[1]) - max(rA[0], rB[0])) / max(1e-9, min(rA[1] - rA[0], rB[1] - rB[0]))
    return mA, mB, round(e1 - e0), round(float(max(dab, dba)), 1), round(float(ov), 2)


def pairs(P=None, save=True):
    P = P or passes()
    nodes, ways = load_osm()
    names = sorted({t["name"] for t, _ in ways if "name" in t and t.get("highway") in O.ROADS})
    XY = {k: O.xy(f.Latitude.values, f.Longitude.values) for k, (_, f) in P.items()}
    rows, masks = [], {}
    for nm in names:
        best = {}
        for ch in chains(nodes, ways, nm):
            if len(ch) < 2: continue
            line = O.xy(*np.array([nodes[x] for x in ch]).T)
            fol = [k for k in P if (O.project(XY[k], line)[0] <= RAD).sum() >= MIN_FOLLOW]
            if len(fol) < 2: continue
            ids, xs_s = xs_on(nodes, ways, nm, ch, line)
            for i, a in enumerate(fol):
                for b in fol[i + 1:]:
                    if a[0] == "i" and b[0] == "i":
                        continue                       # iPhone-iPhone already analysed
                    c = corridor(P[a][1], P[b][1], line, xs_s)
                    if c is None: continue
                    mA, mB, L, lat, ov = c
                    if L > best.get((a, b), (0,))[0]:
                        best[(a, b)] = (L, mA, mB, lat, ov)
        for (a, b), (L, mA, mB, lat, ov) in best.items():
            cat = "S" if (a[0] != b[0] and lat < 5 and ov >= 0.8) else "R"
            rows.append(dict(street=nm, a=a, b=b, corridor_m=L, lateral_med_m=lat, overlap=ov, category=cat,
                             rows_a=int(mA.sum()), rows_b=int(mB.sum())))
            masks[(nm, a, b)] = (mA, mB)
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(X.OUT / "CA_pairs.csv", index=False)
        print(df.to_string(index=False))
    return df, masks


def run():
    P = passes()
    df, masks = pairs(P, save=False)
    m = X.A.load_model(X.CKPT, dev)
    from aic2_recal_table import ride
    from aic2_fault_injection import code_indices, usage_hist, js_div
    src = ride(m, str(X.IPHONE / "summer" / "Nst.csv"))
    scal, _ = X.halves(src["n"]); fr = X.calibrate(src["E"], src["D"], scal)
    srcd, _ = X.load_xyz(str(X.IPHONE / "summer" / "Nst.csv"))
    p_src = usage_hist(code_indices(m, srcd[: len(srcd) // 2]), 1024)
    cit = pd.read_csv(X.OUT / "CITY_citadel_ranking.csv").set_index("segment").rho
    used = sorted(set(df.a) | set(df.b))
    S, prm = {}, []
    for k in used:
        d = P[k][0]; E, D = X.raw_ED(m, d, dev); S[k] = X.apply_frozen(E, D, fr)
        if k[0] == "a":
            rho, _ = seg_rho(m, d); js = js_div(usage_hist(code_indices(m, d), 1024), p_src)
        else:
            rho, js = float(cit[int(k[1:])]), 0.0137
        DS.PROMPT[k] = (round(float(rho), 3), round(float(js), 4))
        prm.append(dict(pass_=k, rho_aux=DS.PROMPT[k][0], JS=DS.PROMPT[k][1]))
    pd.DataFrame(prm).to_csv(X.OUT / "CA_prompts.csv", index=False)
    rows = []
    for r in df.itertuples():
        mA, mB = masks[(r.street, r.a, r.b)]
        W = {r.a: O.strict_support(window_mask_from_rows(mA, CB.windows(P[r.a][0]))),
             r.b: O.strict_support(window_mask_from_rows(mB, CB.windows(P[r.b][0])))}
        for c, e in ((r.a, r.b), (r.b, r.a)):
            ic = np.where(W[c] & np.isfinite(S[c]))[0]; ie = np.where(W[e] & np.isfinite(S[e]))[0]
            base = dict(street=r.street, category=r.category, direction=f"{c}->{e}", n_cal=len(ic), n_eval=len(ie))
            if len(ic) < N_C:
                rows.append({**base, "status": "defer (n_cal<99)"}); continue
            sm = np.concatenate([S[c], S[e]]); off = len(S[c])
            th = X.conformal_tau(sm[ic], X.ALPHA)
            d0 = abs(float(np.mean(sm[ie + off] > fr["tau"])) - X.ALPHA) if len(ie) else np.nan
            d1 = abs(float(np.mean(sm[ie + off] > th)) - X.ALPHA) if len(ie) else np.nan
            whole = ("no evaluation windows" if not len(ie) else "accept" if d1 < d0 else "tie" if d1 == d0 else "refuse")
            A, B = split(ie + off)
            if len(A) < MIN_HALF or len(B) < MIN_HALF:
                rows.append({**base, "whole_audit": whole, "status": "too small for split"}); continue
            row = decide(base["direction"], sm, ic, ie + off, fr, c)
            row.pop("context"); row.pop("n_cal")
            rows.append({**base, "whole_audit": whole, "status": "split", **row})
    out = pd.DataFrame(rows); out.to_csv(X.OUT / "CA_decision_split.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(out.to_string(index=False))
    summarize(out)


def summarize(out):
    orig = pd.read_csv(X.OUT / "DS_decision_split.csv").assign(category="orig")
    sp = out[out.status == "split"]
    for lab, d in [("S", sp[sp.category == "S"]), ("R", sp[sp.category == "R"]), ("orig", orig),
                   ("pooled (orig+S+R)", pd.concat([orig, sp], ignore_index=True)),
                   ("orig+R", pd.concat([orig, sp[sp.category == "R"]], ignore_index=True))]:
        if not len(d):
            print(f"[{lab}] no directions"); continue
        u = d.update_on_B
        s = f"[{lab}] n={len(d)} harmful={int((u=='harmful').sum())} beneficial={int((u=='beneficial').sum())} neutral={int((u=='neutral').sum())}"
        for nm, col in [("CER-only", "CERonly_accepts"), ("Full", "Full_accepts")]:
            a = d[col].astype(bool)
            s += f" | {nm}: acc.harm={int((a&(u=='harmful')).sum())} ref.ben={int((~a&(u=='beneficial')).sum())}"
        s += " | mean devB " + str({k: round(d['dev_B_' + k].mean(), 2) for k in ['Never', 'Always', 'CER-only', 'Full', 'Oracle']})
        print(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("gate0", "pairs", "run"): ap.add_argument("--" + f, action="store_true")
    a = ap.parse_args()
    if a.gate0: gate0()
    if a.pairs: pairs()
    if a.run: run()
