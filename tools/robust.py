"""Robustness check of an exported model under input perturbations (relative comparison on a fixed subset).
   python tools/robust.py accurate [n_excerpts]"""
import sys, os, json, numpy as np
from scipy.signal import resample_poly, butter, sosfilt
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tools"))
import evaluate as ev
from kickdetect.kick_detector import KickDetector
mode = sys.argv[1]; nex = int(sys.argv[2]) if len(sys.argv) > 2 else 40
names = [n for n in ev.names(True) if not ev.meta()[n].get("empty")]; rng = np.random.default_rng(0); names = list(rng.choice(names, nex, replace=False))
def pink(n, rng):
    w = rng.standard_normal(n); return sosfilt(butter(1, 200, fs=44100, output="sos"), w) * 3 + w * 0.3
CASES = {
    "reference (44.1 kHz, as mastered)": lambda x, sr: (x, sr, 1.0),
    "gain -30 dB": lambda x, sr: (x * 10 ** (-30 / 20), sr, 1.0),
    "gain +6 dB with hard clipping": lambda x, sr: (np.clip(x * 2, -1, 1), sr, 1.0),
    "resampled to 48 kHz": lambda x, sr: (resample_poly(x, 160, 147), 48000, 1.0),
    "pitched +6% (DJ pitch fader)": lambda x, sr: (resample_poly(x, 100, 106), sr, 1 / 1.06),
    "pitched -6% (DJ pitch fader)": lambda x, sr: (resample_poly(x, 100, 94), sr, 1 / 0.94),
    "low-pass 12 kHz (lossy source)": lambda x, sr: (sosfilt(butter(6, 12000, fs=sr, output="sos"), x), sr, 1.0),
    "high-pass 60 Hz (small PA / club EQ cut)": lambda x, sr: (sosfilt(butter(2, 60, btype="highpass", fs=sr, output="sos"), x), sr, 1.0),
    "added broadband noise at -30 dB": lambda x, sr: (x + 10 ** (-30 / 20) * np.sqrt(np.mean(x ** 2)) * pink(len(x), np.random.default_rng(1)) / 1.0, sr, 1.0),
}
ev.POST = 0.075
only = sys.argv[3] if len(sys.argv) > 3 else None
for label, f in CASES.items():
    if only and only not in label: continue
    tot = np.zeros(3, int)
    for n in names:
        x, sr = ev.load(n); y, sr2, ts = f(x.astype(np.float64), sr)
        det = [e["time"] for e in KickDetector(sr2, mode=mode).process(y) if e["time"] >= 2.0]
        ref = [r * ts for r in json.load(open(os.path.join(HERE, "data/ref", n + ".json"))) if r * ts >= 2.0]
        tp, fp, fn, _ = ev.match(ref, det); tot += (tp, fp, fn)
    tp, fp, fn = tot; print(f"| {label} | {tp/(tp+fp):.3f} | {tp/(tp+fn):.3f} | {2*tp/(2*tp+fp+fn):.3f} |", flush=True)
