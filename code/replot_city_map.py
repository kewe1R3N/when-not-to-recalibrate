"""Redraw fig_city_spatial (titles only changed) without rescoring: same geometry as aic2_city_osm.main()."""
import numpy as np, pandas as pd
import aic2_city_osm as O
from aic2_city import IP, load_citadel_segment
nodes, ways = O.load_osm()
line = O.street_centerline(nodes, ways)
xs = O.intersections(nodes, ways)
xs_pts = O.xy(*np.array([v[0] for v in xs.values()]).T)
xs_d, xs_s, _ = O.project(xs_pts, line)
keep = xs_d < 25
xs_s_all = xs_s[keep]; xs_names = [v[1] for v, k in zip(xs.values(), keep) if k]
xs_ids = [k_ for k_, k in zip(xs.keys(), keep) if k]
maps = {}
for segA, segB in [(6, 16), (3, 12)]:
    _, fA = load_citadel_segment(IP[segA - 1]); _, fB = load_citadel_segment(IP[segB - 1])
    fA = fA.assign(Latitude=pd.to_numeric(fA.Latitude), Longitude=pd.to_numeric(fA.Longitude))
    fB = fB.assign(Latitude=pd.to_numeric(fB.Latitude), Longitude=pd.to_numeric(fB.Longitude))
    mA, mB, ends, pa, pb = O.osm_masks(fA, fB, line, xs_s_all, 15, 0, xs_ids, (2, 98))
    n0_, n1_ = O.osm_masks.last_nodes
    maps[(segA, segB)] = dict(fA=fA, fB=fB, mA=mA, mB=mB, ends=ends, pa=pa, pb=pb,
                              names=(xs_names[xs_ids.index(n0_)], xs_names[xs_ids.index(n1_)]))
O.plot_maps(maps, line, O.sidewalks(*O.load_osm()))
print("ok")
