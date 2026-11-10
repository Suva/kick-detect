"""Causal rhythm prior: comb-filter bank on the kick activation predicts where the next kicks should fall."""
import numpy as np
class CombPrior:
    def __init__(s, fr, bpm_lo=105.0, bpm_hi=215.0, half_life=2.5):
        s.P = np.arange(int(fr * 60 / bpm_hi), int(fr * 60 / bpm_lo) + 1)
        s.alpha = 0.5 ** (s.P / (half_life * fr)); s.L = int(s.P.max()) + 1
        s.buf = np.zeros((len(s.P), s.L), np.float32); s.i = 0
        s.score = np.zeros(len(s.P), np.float32); s.sdec = np.exp(-1 / (3.0 * fr)); s.rows = np.arange(len(s.P)); s.pk = 1e-3; s.pdec = np.exp(-1 / (4.0 * fr))
    def step(s, a):
        """a: activation in [0,1] for this frame. Returns prior in [0,1] computed from the PAST only."""
        prev = s.buf[s.rows, (s.i - s.P) % s.L]                 # comb outputs one period ago
        k = int(np.argmax(s.score)); pred = float(prev[k])
        s.pk = max(s.pk * s.pdec, pred); prior = pred / max(s.pk, 0.05)
        s.score = s.score * s.sdec + a * prev
        s.buf[:, s.i % s.L] = (1 - s.alpha) * a + s.alpha * prev
        s.i += 1
        return prior, 60.0 / s.P[k]
def run(act, fr):
    c = CombPrior(fr); out = np.zeros(len(act), np.float32); per = np.zeros(len(act), np.float32)
    for n, a in enumerate(act): out[n], per[n] = c.step(float(a))
    return out, per
