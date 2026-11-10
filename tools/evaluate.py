"""Evaluation harness: run a detector over the corpus and score against reference annotations."""
import os, sys, json, glob, importlib, collections, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
PRE, POST = 0.030, 0.070   # a detection matches a reference kick if it lies in [ref-PRE, ref+POST]

def match(ref, det):
    ref = sorted(ref); det = sorted(det); used = [False] * len(det); pairs = []; j0 = 0
    for r in ref:
        best = None
        for j in range(j0, len(det)):
            if det[j] < r - PRE: j0 = j + 1; continue
            if det[j] > r + POST: break
            if not used[j]: best = j; break
        if best is not None: used[best] = True; pairs.append((r, det[best]))
    tp = len(pairs); return tp, len(det) - tp, len(ref) - tp, pairs

_cache = {}
def load(name):
    if name not in _cache:
        x, sr = sf.read(os.path.join(HERE, "corpus", name + ".flac"), dtype="float32", always_2d=True)
        _cache[name] = (x.mean(1), sr)
    return _cache[name]

def meta():
    return json.load(open(os.path.join(HERE, "data/ref_meta.json")))
def names(trusted=None):
    n = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(HERE, "data/ref/*.json")))
    if trusted is None: return n
    m = meta(); ov = json.load(open(os.path.join(HERE, "data/ref_overrides.json"))) if os.path.exists(os.path.join(HERE, "data/ref_overrides.json")) else {}
    return [x for x in n if (bool(m.get(x, {}).get("trusted")) and x not in ov) == trusted]

def run(detect, verbose=True, only=None, refdir="data/ref"):
    tot = collections.defaultdict(lambda: np.zeros(3, int)); lat = []; rows = []; dets = {}
    for name in names():
        if only and not any(o in name for o in only): continue
        ref = json.load(open(os.path.join(HERE, refdir, name + ".json")))
        x, sr = load(name)
        det = detect(x, sr); dets[name] = det
        tp, fp, fn, pairs = match(ref, det)
        style = name.split("_")[0]
        tot[style] += (tp, fp, fn); tot["ALL"] += (tp, fp, fn)
        lat += [d - r for r, d in pairs]; rows.append((name, tp, fp, fn))
    if verbose:
        for s, (tp, fp, fn) in sorted(tot.items()):
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            print(f"{s:11s} TP={tp:5d} FP={fp:4d} FN={fn:4d}  P={p:.3f} R={r:.3f} F={2*p*r/max(p+r,1e-9):.3f}")
        if lat: print(f"latency ms: median={1e3*np.median(lat):.1f} p10={1e3*np.percentile(lat,10):.1f} p90={1e3*np.percentile(lat,90):.1f}")
        worst = sorted(rows, key=lambda r: -(r[2] + r[3]))[:15]
        print("worst:", "  ".join(f"{n}(fp{fp},fn{fn})" for n, tp, fp, fn in worst))
    return tot, rows, dets

if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1]); run(mod.detect, only=sys.argv[2:] or None)
