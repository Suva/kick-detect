"""End-to-end check of the shipped detector on tracks never used for training or cross-validation.
   Annotates corpus/holdout_* with the offline reference and scores kickdetect/kick_detector.py on them."""
import sys, os, glob, json, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tools"))
import evaluate as ev, reference as R
from kickdetect.kick_detector import KickDetector
names = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(HERE, "corpus/holdout_*.flac")))
cache = os.path.join(HERE, "data/holdout_ref.json")
if os.path.exists(cache): REF = json.load(open(cache))
else:
    REF = {}
    for n in names:
        k, info = R.annotate_best(n); REF[n] = dict(k=k, info=info)
    json.dump(REF, open(cache, "w"))
tr = [n for n in names if REF[n]["info"].get("trusted")]
print(f"hold-out excerpts {len(names)} from {len({n.split('_')[1] for n in names})} tracks; annotation trusted for {len(tr)} ({sum(1 for n in tr if REF[n]['info'].get('empty'))} kick-free)")
for mode, post in (("accurate", 0.07), ("fast", 0.07)):
    for ta in (True, False):
        ev.POST = post; tot = np.zeros(3, int); neg = 0; lat = []; per = {}
        for n in tr:
            x, sr = ev.load(n); det = [e["time"] for e in KickDetector(sr, mode=mode, tempo_assist=ta).process(x) if e["time"] >= 2.0]
            ref = [r for r in REF[n]["k"] if r >= 2.0]; tp, fp, fn, pairs = ev.match(ref, det); per[n] = (tp, fp, fn)
            if REF[n]["info"].get("empty"): neg += fp
            else: tot += (tp, fp, fn); lat += [d - r for r, d in pairs]
            json.dump(det, open(os.path.join(HERE, f"data/det_{n}.json"), "w"))
        ov = json.load(open(os.path.join(HERE, "data/ref_overrides.json")))
        tp, fp, fn = sum(np.array(v) for k, v in per.items() if not REF[k]["info"].get("empty"))
        print(f"| all {len(per)} annotated excerpts incl. 1 with a verified annotation fault: `{mode}`, tempo assist {'on' if ta else 'off'} | {tp+fn} | {fn} ({100*fn/(tp+fn):.1f}%) | {fp} ({100*fp/(tp+fp):.1f}%) | {tp/(tp+fp):.3f} | {tp/(tp+fn):.3f} | {2*tp/(2*tp+fp+fn):.3f} |")
        tp, fp, fn = sum(np.array(v) for k, v in per.items() if not REF[k]["info"].get("empty") and k not in ov)
        print(f"| `{mode}`, tempo assist {'on' if ta else 'off'} | {tp+fn} | {fn} ({100*fn/(tp+fn):.1f}%) | {fp} ({100*fp/(tp+fp):.1f}%) | {tp/(tp+fp):.3f} | {tp/(tp+fn):.3f} | {2*tp/(2*tp+fp+fn):.3f} | {1e3*np.median(lat):.0f} ms | {neg} |", flush=True)
        if mode == "accurate" and ta: print("   worst:", sorted(((v[1] + v[2], k, v) for k, v in per.items()), reverse=True)[:6])
