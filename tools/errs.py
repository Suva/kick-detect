"""Per-excerpt error table from saved CV activations: python errs.py acts.npz lat_ms thr [N]"""
import sys, json, numpy as np
sys.path.insert(0, "tools"); import evaluate as ev, nn as N
acts = np.load(sys.argv[1]); lat = float(sys.argv[2]); thr = float(sys.argv[3]); topn = int(sys.argv[4]) if len(sys.argv) > 4 else 30
ev.POST = max(0.07, lat / 1000 + 0.035); names = [n for n in ev.names(True) if n in acts]
A = {n: (acts[n], 501.136) for n in names}; tot, per, l = N.score(A, thr, names)
rows = sorted(per.items(), key=lambda kv: -(kv[1][1] + kv[1][2])); E = sum(v[1] + v[2] for v in per.values())
tp, fp, fn = tot["ALL"]; print(f"ALL TP={tp} FP={fp} FN={fn} F={2*tp/(2*tp+fp+fn):.4f}; total errors {E}; top{topn} excerpts carry {sum(v[1]+v[2] for k,v in rows[:topn])}; excerpts with <=2 errors: {sum(1 for v in per.values() if v[1]+v[2]<=2)}/{len(per)}")
m = ev.meta()
for k, v in rows[:topn]: print(f"{k:22s} tp={v[0]:3d} fp={v[1]:3d} fn={v[2]:3d}  bpm={m[k].get('bpm')} rho={m[k].get('rho')} rho10={m[k].get('rho10')} on={m[k].get('ongrid')}/{m[k].get('beats')} n={m[k].get('n')}")
