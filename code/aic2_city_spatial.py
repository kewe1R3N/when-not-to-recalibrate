#!/usr/bin/env python3
"""
Spatial-selection robustness for the Charleston cross-pass test.

The reported test (aic2_city.py, L2) keeps rows whose raw lat/lon, rounded to
four decimals (~11 m cells), occurs in both passes. This script re-runs the
same test with everything held fixed (frozen model, Lincoln source
normalization, window-span rule, full-smoothing-support trim, split-conformal
rule) and only the spatial-selection rule changed:

  R0  rounded-cell intersection, 4 decimals (reproduces the reported counts)
  R1  centerline projection: both passes are projected onto one centerline
      (the smoothed trajectory of the pass with more GPS fixes); a row is kept
      if its lateral offset is <= 15 m and its chainage lies inside the
      chainage interval covered by BOTH passes (fixed corridor endpoints)
  R2  as R1 with a 25 m lateral radius

R1/R2 are the single-street geometry that road-network map matching would
produce (snap to a line, compare along-line position). No road network is
used, because none for Charleston is available offline; this is a
trajectory-centerline projection, not HMM map matching.
"""
import numpy as np, pandas as pd, torch
import aic2_exp as X
import aic2_codebook as CB
from aic2_city import (IP, load_citadel_segment, window_mask_from_rows,
                       trim_to_full_smoothing_support, dev)

ALPHA = X.ALPHA
N_C = int(np.ceil(1 / ALPHA)) - 1          # 99 at alpha = 0.01


def enu(lat, lon, lat0, lon0):
    R = 6371000.0
    x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R
    return np.c_[x, y]


def centerline(pts, k=5):
    """Ordered unique fixes, moving-average smoothed, as a polyline."""
    _, idx = np.unique(pts, axis=0, return_index=True)
    p = pts[np.sort(idx)]
    if len(p) > 2 * k:
        ker = np.ones(k) / k
        p = np.c_[np.convolve(p[:, 0], ker, "valid"), np.convolve(p[:, 1], ker, "valid")]
    return p


def project(pts, line):
    """Lateral distance and chainage of each point on a polyline."""
    a, b = line[:-1], line[1:]
    ab = b - a
    seglen = np.linalg.norm(ab, axis=1)
    cum = np.r_[0, np.cumsum(seglen)]
    best_d = np.full(len(pts), np.inf); best_s = np.zeros(len(pts))
    for i in range(len(a)):
        t = np.clip(((pts - a[i]) @ ab[i]) / max(seglen[i] ** 2, 1e-9), 0, 1)
        q = a[i] + t[:, None] * ab[i]
        d = np.linalg.norm(pts - q, axis=1)
        upd = d < best_d
        best_d[upd] = d[upd]; best_s[upd] = cum[i] + t[upd] * seglen[i]
    return best_d, best_s


def masks_cells(fA, fB, dec=4):
    cA = list(zip(fA["Latitude"].round(dec), fA["Longitude"].round(dec)))
    cB = list(zip(fB["Latitude"].round(dec), fB["Longitude"].round(dec)))
    sh = set(cA) & set(cB)
    return np.array([c in sh for c in cA]), np.array([c in sh for c in cB])


def masks_centerline(fA, fB, radius):
    la = pd.to_numeric(fA["Latitude"], errors="coerce").values
    lo = pd.to_numeric(fA["Longitude"], errors="coerce").values
    lb = pd.to_numeric(fB["Latitude"], errors="coerce").values
    ob = pd.to_numeric(fB["Longitude"], errors="coerce").values
    lat0, lon0 = np.nanmean(la), np.nanmean(lo)
    pA, pB = enu(la, lo, lat0, lon0), enu(lb, ob, lat0, lon0)
    ref = pA if len(np.unique(pA, axis=0)) >= len(np.unique(pB, axis=0)) else pB
    line = centerline(ref[~np.isnan(ref).any(1)])
    dA, sA = project(np.nan_to_num(pA), line)
    dB, sB = project(np.nan_to_num(pB), line)
    onA, onB = dA <= radius, dB <= radius
    lo_s = max(sA[onA].min(), sB[onB].min()); hi_s = min(sA[onA].max(), sB[onB].max())
    mA = onA & (sA >= lo_s) & (sA <= hi_s) & ~np.isnan(pA).any(1)
    mB = onB & (sB >= lo_s) & (sB <= hi_s) & ~np.isnan(pB).any(1)
    return mA, mB, hi_s - lo_s


def main():
    m = X.A.load_model(X.CKPT, dev)
    src, _ = X.load_xyz(str(X.IPHONE / "summer" / "Nst.csv"))
    n0 = CB.windows(src); E0, D0 = X.raw_ED(m, src, dev)
    cal0 = np.zeros(n0, bool); cal0[: n0 // 2] = True
    fr = X.calibrate(E0, D0, cal0)

    rules = {"R0 rounded cells (4 dp)": lambda a, b: masks_cells(a, b) + (np.nan,),
             "R1 centerline, 15 m": lambda a, b: masks_centerline(a, b, 15.0),
             "R2 centerline, 25 m": lambda a, b: masks_centerline(a, b, 25.0)}
    rows = []
    for segA, segB in [(6, 16), (3, 12)]:
        dA, fA = load_citadel_segment(IP[segA - 1]); dB, fB = load_citadel_segment(IP[segB - 1])
        EA, DA = X.raw_ED(m, dA, dev); EB, DB = X.raw_ED(m, dB, dev)
        smA, smB = X.apply_frozen(EA, DA, fr), X.apply_frozen(EB, DB, fr)
        nA, nB = CB.windows(dA), CB.windows(dB)
        for rname, rule in rules.items():
            mA, mB, span = rule(fA, fB)
            wA = trim_to_full_smoothing_support(window_mask_from_rows(mA, nA))
            wB = trim_to_full_smoothing_support(window_mask_from_rows(mB, nB))
            for (ta, sa, wa, tb, sb, wb) in [(segA, smA, wA, segB, smB, wB),
                                             (segB, smB, wB, segA, smA, wA)]:
                ncal = int(wa.sum())
                frozen = float(np.mean(sb[wb] > fr["tau"])) if wb.sum() else np.nan
                if ncal >= N_C:
                    tau = X.conformal_tau(sa[wa], ALPHA)
                    conf = float(np.mean(sb[wb] > tau)); status = "finite"
                else:
                    tau, conf, status = np.inf, 0.0, "insufficient"
                rows.append(dict(rule=rname, direction=f"{ta}->{tb}", n_cal=ncal,
                                 n_eval=int(wb.sum()), overlap_m=round(span, 0),
                                 tau=tau if np.isfinite(tau) else "inf",
                                 CER_frozen=round(frozen, 4), CER_conformal=round(conf, 4),
                                 status=status))
    df = pd.DataFrame(rows)
    df.to_csv(X.OUT / "CITY_spatial_robustness.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))
    print("\nfull-segment windows (before any spatial rule):",
          {s: CB.windows(load_citadel_segment(IP[s - 1])[0]) for s in (6, 16, 3, 12)})


if __name__ == "__main__":
    main()
