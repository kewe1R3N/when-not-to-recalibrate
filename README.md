# When Not to Recalibrate — data and code

Data and analysis code for

> K. Ren and C.-H. Ho, *When Not to Recalibrate: Reference-Corridor Auditing of a Frozen
> Smartphone–Bicycle Pavement Monitor across Seasons, Devices, and Cities*
> (submitted to *Automation in Construction*).

## Contents

```
data/
  lincoln_iphone/{2023-10,2023-11_12,2024-01,2024-05}/{11th,13th,16th,Nst,billy}.csv
  lincoln_android/{11th,16th,Nst,billy}.csv
  charleston_citadel_iphone/segment_01..20.csv
  charleston_ravenel_android/{TotalAcceleration,Gravity}.csv
  columbia_iphone/{R1,R2a,R2b}.csv
corridors.csv        OpenStreetMap node IDs that fix each corridor's ends
MANIFEST.csv         every data file and its row count
results/             result tables behind the manuscript and supplement
code/                analysis, fault-injection, and figure scripts
checkpoint/          frozen VQTransAE win-65 weights (1,848,367 parameters)
```

The 2023-10 campaign has no 11th St ride, and Android has no 13th St ride.
`billy` is the Billy Wolff Trail (concrete).

## Location privacy: GPS is marked, not published

Latitude, longitude, absolute timestamps, device identifiers, and auxiliary
phone channels (heart rate, sound level, magnetometer, and so on) were removed.
Instead, each row of a ride file carries:

| column | meaning |
|---|---|
| `t_s` | time since the start of the file (s); from the logged clock where it parses, otherwise row index / 10 Hz |
| `X, Y, Z` | acceleration as logged (Lincoln iPhone and Citadel: g, gravity present; Lincoln Android: already standardized, gravity removed) |
| `gps_fix` | 1 if the row had a valid position fix, else 0 |
| `along_track_m` | cumulative distance travelled since the first fix (m) |
| `route_dist_m` | Lincoln Android only: upstream map-matched distance along the route (m) |
| `chainage_m` | analysed corridors only: distance along the OpenStreetMap axis from the corridor origin node (m) |
| `offset_m` | analysed corridors only: signed lateral offset from the mapped line (m) |
| `in_corridor` | analysed corridors only: 1 if the row lies inside the corridor used in the paper |

The analysed corridors are N St in Lincoln (`Nst.csv`, all campaigns and Android),
East Bay Street in Charleston (segments 3, 6, 12, and 16), and Main Street and
Sumter Street in Columbia (passes R2a and R2b; columns prefixed `main_` and
`sumter_`). Columbia R2a and R2b are two separate rides of one campus loop,
15 minutes apart in the same direction; R1 is a different loop.
`corridors.csv` gives the OSM node IDs at each corridor's ends, so the chainage
axes can be rebuilt from OpenStreetMap. The N St axis uses the principal axis of
the N Street roadway with its origin at node 134968191; the East Bay axis runs
along the street's mapped polyline from south to north.

The Ravenel Bridge ride is the raw 105 Hz Sensor Logger export, reduced to
`seconds_elapsed, x, y, z` in m/s² for total acceleration and gravity.

## What this release reproduces

- All threshold results (fixed-threshold matrix, recalibration table,
  Charleston cross-pass test, corridor-length curve), by rescoring the rides
  with the checkpoint.
- The fault-injection and ablation results, by re-injecting the six faults
  into `lincoln_iphone/2024-05/Nst.csv` (`code/aic2_fault_injection.py`,
  `code/protocol_ablation.py`).
- The Columbia repeated-pass test (`code/aic2_columbia.py`; the pre-specified
  corridor rule and its logged deviations are in Supplementary Section S12).
- The spatial-repeatability analysis (`code/aic2_chainage*.py`), from the
  `chainage_m` and `offset_m` columns.

**Not included:**
- The study-area and Charleston satellite maps, which need raw coordinates.
- The video-annotation labels behind the PR-AUC comparison (Supplementary
  Section S7), which belong to the earlier VQTransAE dataset.

## Running the code

The scripts were written against the authors' directory layout. Set the paths at
the top of `code/aic2_exp.py`:
- `CODE`, `IPHONE`, `ANDROID_NST` and `CKPT` point to `data/` and `checkpoint/`.
- `OUT` points to an output directory.

Loaders that read latitude/longitude must use the `chainage_m` / `in_corridor`
columns instead. Python 3.10, PyTorch, NumPy, pandas, SciPy and Matplotlib are
required.

## Licence

Code: MIT. Data and checkpoint: CC BY 4.0. Street geometry referenced through
OpenStreetMap node IDs © OpenStreetMap contributors, ODbL.

## Acknowledgments

Part of this study was supported by the Center for Connected Multimodal Mobility
(C2M2), a USDOT Tier 1 University Transportation Center.
