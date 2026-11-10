"""IIR comb prior (tools/rhythm.py) variants incl. re-entry recall. python rhythm_eval3.py acts.npz lat_ms"""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, "tools"); import evaluate as ev, nn as N, rhythm
acts = np.load(sys.argv[1]); lat_ms = float(sys.argv[2]); ev.POST = max(0.07, lat_ms / 1000 + 0.035)
names = [n for n in ev.names(True) if n in acts]; fr = 501.136
cache = sys.argv[1] + ".iir.pkl"
if os.path.exists(cache): PR = pickle.load(open(cache, "rb"))
else:
    PR = {n: rhythm.run(acts[n], fr)[0] for n in names}; pickle.dump(PR, open(cache, "wb"))
PR = {n: PR[n] for n in names if n in PR}; names = [n for n in names if n in PR]
def logit(a): a = np.clip(a, 1e-6, 1 - 1e-6); return np.log(a / (1 - a))
def reentry(A, thr):
    hit = tot = 0
    for n in names:
        ref = [r for r in json.load(open(f"data/ref/{n}.json")) if r >= 2.0]; det = np.array(N.pick(A[n][0], fr, thr))
        for i, r in enumerate(ref):
            if i > 0 and r - ref[i - 1] >= 1.5: tot += 1; hit += int(len(det) and np.any((det >= r - ev.PRE) & (det <= r + ev.POST)))
    return hit, tot
def run(f, label):
    best = None
    for thr in (0.7, 0.8, 0.9, 0.95, 0.97, 0.985, 0.993, 0.997):
        A = {n: (1 / (1 + np.exp(-f(logit(acts[n]), PR[n]))), fr) for n in names}
        tot, per, lat = N.score(A, thr, names); tp, fp, fn = tot["ALL"]; F = 2 * tp / max(2 * tp + fp + fn, 1)
        if best is None or F > best[0]: best = (F, thr, tp, fp, fn, A, tot)
    h, t = reentry(best[5], best[1])
    print(f"{label:28s} F={best[0]:.4f} thr={best[1]} TP={best[2]} FP={best[3]} FN={best[4]} P={best[2]/(best[2]+best[3]):.3f} R={best[2]/(best[2]+best[4]):.3f} re-entry {h}/{t}", flush=True)
    return best
run(lambda z, p: z, "baseline")
for b in (2.0, 3.0): run(lambda z, p, b=b: z + b * (2 * p - 1), f"sym beta={b}")
for b in (2.0, 3.0, 4.0): run(lambda z, p, b=b: z + b * p, f"boost beta={b}")
for b in (3.0,): run(lambda z, p, b=b: z + b * (2 * p - 1) * (p > 0.5) , f"boost-only-above-half beta={b}")
for b in (2.0, 3.0): run(lambda z, p, b=b: z + b * (1.5 * p - 0.5), f"asym(-0.5..+1) beta={b}")
