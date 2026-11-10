"""Create port-verification vectors: a 12 s clip plus expected activations/events for each weight set.
   python tests/make_vectors.py [corpus_excerpt_name]"""
import os, sys, json, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from kickdetect.kick_detector import KickDetector
name = sys.argv[1] if len(sys.argv) > 1 else "hardtechno_e4dbf109_b"
x, sr = sf.read(os.path.join(HERE, "corpus", name + ".flac"), dtype="float32", always_2d=True)
x = x.mean(1)[8 * sr:20 * sr].astype(np.float32)
V = os.path.join(HERE, "tests/vectors"); x.tofile(os.path.join(V, "clip.f32"))
json.dump(dict(sample_rate=sr, samples=len(x), format="raw mono float32 little-endian", source=name), open(os.path.join(V, "clip.json"), "w"), indent=1)
for mode in ("accurate", "fast"):
    if not os.path.exists(os.path.join(HERE, "kickdetect", f"weights_{mode}.json")): continue
    for ta in (True, False):
        d = KickDetector(sr, mode=mode, tempo_assist=ta); ev, acts = d.process(x, return_activations=True)
        tag = f"{mode}_{'tempo' if ta else 'notempo'}"
        open(os.path.join(V, f"activations_{tag}.txt"), "w").write("\n".join(f"{a:.6f}" for a in acts) + "\n")
        open(os.path.join(V, f"events_{tag}.txt"), "w").write("".join(f"{e['time']:.4f}\t{e['onset_time']:.4f}\t{e['strength']:.3f}\n" for e in ev))
        print(tag, "frames", len(acts), "events", len(ev))
