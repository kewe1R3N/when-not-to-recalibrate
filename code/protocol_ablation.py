"""Protocol ablation: what each review layer adds, on the frozen case set.

Inputs are the already-recorded result files; nothing is re-run and no number
is invented here. The only judgement encoded is DIRECT_CHECK_DETECTABLE, which
records whether a declared-versus-actual acquisition check can see each fault;
its justification is stated per case and repeated in the paper.
"""
import csv, json, pathlib

R = pathlib.Path(__file__).parent
ALPHA = 0.01
RHO_LIMIT, JS_LIMIT = 0.80, 0.044

grid = {r["fault"]: r for r in csv.DictReader(open(R / "FI_fault_grid.csv"))}
jsctx = {r["context"]: r for r in csv.DictReader(open(R / "FI_js_context.csv"))}
rho_nst = {r["camp"]: float(r["rho"]) for r in csv.DictReader(open(R / "M1cv2_ranking_allroutes.csv"))
           if r["route"] == "Nst"}

# a direct check compares the declared acquisition convention with what the
# stream discloses; it sees a fault only when the fault changes something declared
DIRECT_CHECK_DETECTABLE = {
    "unit_ms2":    (True,  "magnitude scale contradicts the declared g units"),
    "gravity_rm":  (True,  "the ~1 g component on the gravity axis is absent"),
    "missing_std": (True,  "the standardization step is pipeline state and is checkable"),
    "axswap_YZ":   (True,  "gravity moves off Z onto Y"),
    "axswap_XY":   (False, "gravity stays on Z; no declared quantity changes"),
    "rate_2x":     (True,  "sample count contradicts the declared 10 Hz timebase"),
}

def dev(x): return abs(float(x) - ALPHA)

rows = []
for f, r in grid.items():
    if f == "none":
        continue
    frozen, conf = float(r["FPR_frozen"]), float(r["FPR_conformal"])
    rho, js = float(r["rho_corridor"]), float(r["JS_usage"])
    direct, why = DIRECT_CHECK_DETECTABLE[f]
    rows.append(dict(case=f, truth="input fault", cls=r["cls"],
                     frozen=frozen, conf=conf,
                     adverse=dev(conf) > dev(frozen),
                     direct=direct, why=why,
                     rho=rho, rho_flag=rho < RHO_LIMIT,
                     js=js, js_flag=js > JS_LIMIT))

CLEAN_RHO = {"noise_floor(clean 2nd half)": rho_nst["summer"], "autumn": rho_nst["autumn"],
             "spring": rho_nst["spring"], "winter": rho_nst["winter"], "android": rho_nst["android"],
             "citadel_iphone(city)": 0.923, "ravenel(city+device+app)": 0.852}
# frozen / conformal corridor CER on the held-out target block (summer source)
CLEAN_CER = {"autumn": (0.045, 0.000), "winter": (0.039, 0.000), "spring": (0.043, 0.016),
             "android": (0.077, 0.101), "noise_floor(clean 2nd half)": (0.043, 0.043)}
for c, r in jsctx.items():
    js = float(r["JS"]); rho = CLEAN_RHO[c]
    frozen, conf = CLEAN_CER.get(c, (None, None))
    rows.append(dict(case=c, truth="clean", cls=r["kind"], frozen=frozen, conf=conf,
                     adverse=(dev(conf) > dev(frozen)) if frozen is not None else None,
                     direct=False, why="conventions match what is declared",
                     rho=rho, rho_flag=rho < RHO_LIMIT, js=js, js_flag=js > JS_LIMIT))

def tally(layer):
    missed = ft = unres = 0
    for r in rows:
        flagged = ((layer >= 2 and r["direct"]) or (layer >= 3 and r["rho_flag"])
                   or (layer >= 4 and r["js_flag"]))
        if r["truth"] == "input fault":
            if flagged: continue
            if r["adverse"]: unres += 1
            else: missed += 1
        else:
            if flagged: ft += 1
            elif r["adverse"]: unres += 1
    return missed, ft, unres

names = {1: "quantile update only", 2: "+ direct input checks",
         3: "+ auxiliary correlation", 4: "+ codebook JS"}
print(f"{'variant':26} {'missed/6':>9} {'false trig/7':>13} {'unresolved':>11}")
summary = {}
for L in (1, 2, 3, 4):
    m, f, u = tally(L); summary[names[L]] = dict(missed=m, false_trigger=f, unresolved=u)
    print(f"{names[L]:26} {m:>9} {f:>13} {u:>11}")

print("\nunique catcher per known fault:")
for r in rows:
    if r["truth"] != "input fault": continue
    who = [n for n, ok in (("direct", r["direct"]), ("rho", r["rho_flag"]), ("JS", r["js_flag"])) if ok]
    print(f"  {r['case']:12} caught by {who if who else 'NOTHING'}")

# leave-one-context-out limit for JS: does the clean false trigger survive?
print("\nJS leave-one-context-out (clean cases):")
for r in rows:
    if r["truth"] != "clean": continue
    others = [float(x["JS"]) for k, x in jsctx.items() if k != r["case"]]
    loo = max(others)
    print(f"  {r['case']:28} JS={r['js']:.4f}  LOO limit={loo:.4f}  flagged={r['js']>loo}")

json.dump(dict(summary=summary, cases=rows), open(R / "protocol_ablation.json", "w"), indent=2)
with open(R / "protocol_ablation.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
