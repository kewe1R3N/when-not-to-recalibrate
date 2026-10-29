#!/usr/bin/env python3
"""Out-of-sample decision quality (refine-logs/PREREG_decision_split.md):
decide on the first half of each audit set, evaluate on the second half."""
import numpy as np, pandas as pd, torch
import aic2_exp as X
import aic2_codebook as CB
import aic2_city_osm as O
from aic2_recal_table import ride
from aic2_city import IP, load_citadel_segment, window_mask_from_rows
import aic2_columbia as COL

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A_ = X.ALPHA; G = 12
PROMPT = {"Oct": (0.913, 0.0165), "Nov-Dec": (0.959, 0.0441), "Jan": (0.924, 0.0386), "Android": (0.829, 0.0394),
          "C6": (0.94, 0.0137), "C16": (0.90, 0.0137), "C3": (0.90, 0.0137), "C12": (0.91, 0.0137),
          "R2a": (0.867, 0.0507), "R2b": (0.815, 0.0573)}


def split(idx):
    idx = np.sort(idx); h = len(idx) // 2
    return idx[:max(0, h - G)], idx[h + G:]


def dv(sm, idx, tau):
    return abs(float(np.mean(sm[idx] > tau)) - A_) if np.isfinite(tau) else A_


def decide(name, sm, cal_idx, aud_idx, fr, prompt_key):
    t0 = fr["tau"]; th = X.conformal_tau(sm[cal_idx], A_)
    A, B = split(aud_idx)
    dA0, dA1, dB0, dB1 = dv(sm, A, t0), dv(sm, A, th), dv(sm, B, t0), dv(sm, B, th)
    rho, js = PROMPT[prompt_key]; hold = (rho < 0.80) or (js > 0.044)
    cer_only = th if (np.isfinite(th) and dA1 < dA0) else t0
    full = t0 if hold else cer_only
    choose = {"Never": t0, "Always": th if np.isfinite(th) else t0, "CER-only": cer_only, "Full": full,
              "Oracle": th if dB1 < dB0 else t0}
    row = dict(context=name, n_cal=len(cal_idx), n_A=len(A), n_B=len(B),
               CER_A_frozen=round(float(np.mean(sm[A] > t0)), 4), CER_A_cand=round(float(np.mean(sm[A] > th)), 4),
               CER_B_frozen=round(float(np.mean(sm[B] > t0)), 4), CER_B_cand=round(float(np.mean(sm[B] > th)), 4),
               dB_frozen=round(100 * dB0, 2), dB_cand=round(100 * dB1, 2),
               update_on_B=("harmful" if dB1 > dB0 else "beneficial" if dB1 < dB0 else "neutral"),
               CERonly_accepts=bool(cer_only == th), hold=bool(hold), Full_accepts=bool(full == th))
    for k, t in choose.items():
        row["dev_B_" + k] = round(100 * dv(sm, B, t), 2)
    return row


def main():
    m = X.A.load_model(X.CKPT, dev)
    src = ride(m, str(X.IPHONE / "summer" / "Nst.csv"))
    scal, _ = X.halves(src["n"]); fr = X.calibrate(src["E"], src["D"], scal)
    rows = []
    # Lincoln
    for key, ctx in [("Oct", "autumn"), ("Nov-Dec", "winter"), ("Jan", "spring"), ("Android", "android")]:
        r = ride(m, str(X.ANDROID_NST if ctx == "android" else X.IPHONE / ctx / "Nst.csv"))
        cal, tst = X.halves(r["n"]); sm = X.apply_frozen(r["E"], r["D"], fr)
        ok = np.isfinite(sm)
        rows.append(decide(key, sm, np.where(cal & ok)[0], np.where(tst & ok)[0], fr, key))
    # Charleston (primary street-matched rule)
    nodes, ways = O.load_osm(); line = O.street_centerline(nodes, ways)
    xs = O.intersections(nodes, ways); xs_pts = O.xy(*np.array([v[0] for v in xs.values()]).T)
    xs_d, xs_s, _ = O.project(xs_pts, line); keep = xs_d < 25
    xs_s_all = xs_s[keep]; xs_ids = [k for k, kk in zip(xs.keys(), keep) if kk]
    for a, b in [(6, 16), (3, 12)]:
        dA, fA = load_citadel_segment(IP[a - 1]); dB, fB = load_citadel_segment(IP[b - 1])
        fA = fA.assign(Latitude=pd.to_numeric(fA.Latitude), Longitude=pd.to_numeric(fA.Longitude))
        fB = fB.assign(Latitude=pd.to_numeric(fB.Latitude), Longitude=pd.to_numeric(fB.Longitude))
        mA, mB, *_ = O.osm_masks(fA, fB, line, xs_s_all, 15, 0, xs_ids, (2, 98))
        S = {}
        for k, d, mk in ((a, dA, mA), (b, dB, mB)):
            E, D = X.raw_ED(m, d, dev); sm = X.apply_frozen(E, D, fr)
            w = O.strict_support(window_mask_from_rows(mk, CB.windows(d)))
            S[k] = (sm, np.where(w & np.isfinite(sm))[0])
        for c, e in ((a, b), (b, a)):
            # calibration on pass c, audit on pass e (two scored rides): stack into one index space
            smc, ic = S[c]; sme, ie = S[e]
            sm = np.concatenate([smc, sme]); off = len(smc)
            rows.append(decide(f"{c}->{e}", sm, ic, ie + off, fr, f"C{c}"))
    # Columbia Main Street
    nodes, ways = COL.load_osm(); R = {k: COL.load(k) for k in ("R2a", "R2b")}
    ch, line = COL.street_line(nodes, ways, "Main Street", "113980212", "2695777079")
    ids, xs_s2, _ = COL.xs_on(nodes, ways, "Main Street", ch, line)
    mA, mB, *_ = COL.masks(R["R2a"][1], R["R2b"][1], line, xs_s2, ids)
    S = {}
    for k, mk in (("R2a", mA), ("R2b", mB)):
        d = R[k][0]; E, D = X.raw_ED(m, d, dev); sm = X.apply_frozen(E, D, fr)
        w = O.strict_support(window_mask_from_rows(mk, CB.windows(d)))
        S[k] = (sm, np.where(w & np.isfinite(sm))[0])
    for c, e in (("R2a", "R2b"), ("R2b", "R2a")):
        smc, ic = S[c]; sme, ie = S[e]; sm = np.concatenate([smc, sme])
        rows.append(decide(f"Main {c}->{e}", sm, ic, ie + len(smc), fr, c))
    df = pd.DataFrame(rows); df.to_csv(X.OUT / "DS_decision_split.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df.to_string(index=False))
        print("\nmean deviation on B (pp):", {k: round(df["dev_B_" + k].mean(), 2) for k in ["Never", "Always", "CER-only", "Full", "Oracle"]})
        print("updates harmful on B:", int((df.update_on_B == "harmful").sum()), " beneficial:", int((df.update_on_B == "beneficial").sum()),
              " neutral:", int((df.update_on_B == "neutral").sum()))
        for s, col in [("CER-only", "CERonly_accepts"), ("Full", "Full_accepts")]:
            acc = df[col]
            print(s, "accepted harmful:", int((acc & (df.update_on_B == "harmful")).sum()),
                  " refused/held beneficial:", int((~acc & (df.update_on_B == "beneficial")).sum()),
                  " accepted beneficial:", int((acc & (df.update_on_B == "beneficial")).sum()),
                  " refused harmful:", int((~acc & (df.update_on_B == "harmful")).sum()))


if __name__ == "__main__":
    main()
