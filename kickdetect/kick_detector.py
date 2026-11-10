"""Real-time kick drum detector for dance music — Python reference implementation (numpy + scipy).

Streaming usage (any block size, mono float samples in [-1, 1]):

    det = KickDetector(sample_rate=44100)
    for block in audio_blocks:
        for ev in det.process(block):
            print(ev["time"], ev["onset_time"], ev["strength"])

`time` is when the kick is reported (seconds since start); `onset_time` is the estimated time of the
kick's onset itself (time - latency). The same algorithm is implemented in C in kick_detector.h; both read
the weights exported by tools/export.py and produce identical activations (see tests/).
"""
import json
import math
import os

import numpy as np
from scipy.signal import lfilter

FC = [45, 64, 90, 128, 180, 256, 360, 512, 720, 1024, 2048, 4096, 8192]   # band centres (Hz)
QS = [2.0] * 9 + [1.0] * 4                                                # Q of each of the two cascaded biquads
NB = len(FC)
HOP_HZ = 500.0          # feature frames per second (hop = round(sr / 500) samples)
TAU = 4.0               # decay time constant of the slow peak followers (s)
EPS = 1e-3              # level floor relative to the global peak (-60 dB)
REFRACTORY = 0.060      # minimum time between two reported kicks (s)
KT = 5                  # conv taps in time; layers use dilation 1, 2, 4 -> receptive field 29 frames (58 ms)


class KickDetector:
    def __init__(self, sample_rate, mode="accurate", weights=None, threshold=None, tempo_assist=True):
        """mode: "accurate" (~30 ms latency) or "fast" (~14 ms latency); or pass an explicit weights JSON path."""
        path = weights or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"weights_{mode}.json")
        w = json.load(open(path))
        L = {k: np.array(v["data"], np.float32).reshape(v["shape"]) for k, v in w["layers"].items()}
        self.w1, self.b1 = L["c1.weight"], L["c1.bias"]          # (C1, 2, 5, 3)
        self.w2, self.b2 = L["c2.weight"], L["c2.bias"]          # (C2, C1, 5, 3)
        self.w3, self.b3 = L["c3.weight"], L["c3.bias"]          # (C3, C2, 5, 3)
        self.wd1, self.bd1 = L["d1.weight"], L["d1.bias"]        # (HID, C3*7 + 13)
        self.wd2, self.bd2 = L["d2.weight"][0], float(L["d2.bias"][0])
        self.lat_frames = int(w["lat_frames"])
        self.tempo_assist = bool(tempo_assist)
        self.beta = float(w.get("tempo_beta", 3.0))
        default_thr = w.get("threshold_tempo", w["threshold"]) if self.tempo_assist else w["threshold"]
        self.threshold = float(default_thr if threshold is None else threshold)

        self.sr = float(sample_rate)
        self.hop = int(round(self.sr / HOP_HZ))
        self.frame_rate = self.sr / self.hop
        self.latency = self.lat_frames / self.frame_rate
        # --- front-end state
        self.coef = []
        for fc, q in zip(FC, QS):
            w0 = 2 * math.pi * fc / self.sr
            al = math.sin(w0) / (2 * q)
            a0 = 1 + al
            self.coef.append((np.array([al / a0, 0.0, -al / a0]), np.array([1.0, -2 * math.cos(w0) / a0, (1 - al) / a0])))
        self.zi = [[np.zeros(2), np.zeros(2)] for _ in range(NB)]
        self.hold = [np.zeros(int(math.ceil(0.5 / (fc * self.hop / self.sr)))) for fc in FC]
        self.hpos = [0] * NB
        self.pending = np.zeros((NB, 0))          # rectified band signals not yet forming a full hop
        self.dec = math.exp(-1.0 / (TAU * self.frame_rate))
        self.g = 1e-4
        self.r = np.full(NB, 1e-4)
        # --- network state (ring-free: short histories as arrays, newest last)
        self.x0 = np.zeros((KT, 2, NB), np.float32)
        self.h1 = np.zeros((2 * (KT - 1) + 1, self.w1.shape[0], NB - 2), np.float32)
        self.h2 = np.zeros((4 * (KT - 1) + 1, self.w2.shape[0], NB - 4), np.float32)
        # --- tempo assist: comb-filter bank on the activation, one comb per beat period (105..215 BPM)
        fr = self.frame_rate
        self.cP = np.arange(int(fr * 60 / 215), int(fr * 60 / 105) + 1)
        self.cP = self.cP[:160]
        self.cL = int(self.cP[-1]) + 1
        self.calpha = (0.5 ** (self.cP / (2.5 * fr))).astype(np.float32)
        self.cbuf = np.zeros((len(self.cP), self.cL), np.float32)
        self.cscore = np.zeros(len(self.cP), np.float32)
        self.crow = np.arange(len(self.cP))
        self.ci = 0
        self.cpk = 1e-3
        self.csdec = math.exp(-1.0 / (3.0 * fr))
        self.cpdec = math.exp(-1.0 / (4.0 * fr))
        self.tempo_bpm = 0.0
        self.raw_activation = 0.0
        self.nsamples = 0
        self.since_last = 10 ** 9
        self.refractory = int(REFRACTORY * self.frame_rate + 0.5)
        self.activation = 0.0

    # -- network: one new time column per layer ------------------------------------------------
    @staticmethod
    def _conv(hist, w, b):
        """hist: (KT, Cin, W) frames oldest..newest; w: (Cout, Cin, KT, 3). Valid conv over bands, ReLU."""
        Wd = hist.shape[2] - 2
        cols = np.stack([hist[:, :, j:j + Wd] for j in range(3)], -1)          # (KT, Cin, Wd, 3)
        return np.maximum(np.einsum("octj,tcwj->ow", w, cols) + b[:, None], 0.0)

    def _frame(self, F, C):
        self.x0 = np.roll(self.x0, -1, 0); self.x0[-1, 0] = F; self.x0[-1, 1] = F - C
        self.h1 = np.roll(self.h1, -1, 0); self.h1[-1] = self._conv(self.x0, self.w1, self.b1)
        self.h2 = np.roll(self.h2, -1, 0); self.h2[-1] = self._conv(self.h1[::2], self.w2, self.b2)
        h3 = self._conv(self.h2[::4], self.w3, self.b3)
        feat = np.concatenate([h3.ravel(), C])
        z = float(self.wd2 @ np.maximum(self.wd1 @ feat + self.bd1, 0.0) + self.bd2)
        return 1.0 / (1.0 + math.exp(-z))

    def _tempo(self, a):
        """Comb bank step. Returns the activation biased by how well 'now' fits the running kick pattern."""
        prev = self.cbuf[self.crow, (self.ci - self.cP) % self.cL]      # each comb's output one period ago
        k = int(np.argmax(self.cscore))
        pred = float(prev[k])
        self.tempo_bpm = 60.0 * self.frame_rate / float(self.cP[k])
        self.cpk = max(self.cpk * self.cpdec, pred)
        prior = pred / max(self.cpk, 0.05)
        self.cscore = self.cscore * np.float32(self.csdec) + np.float32(a) * prev
        self.cbuf[:, self.ci % self.cL] = (1 - self.calpha) * np.float32(a) + self.calpha * prev
        self.ci += 1
        if not self.tempo_assist:
            return a
        ac = min(max(a, 1e-6), 1 - 1e-6)
        z = math.log(ac / (1 - ac)) + self.beta * (2 * prior - 1)
        return 1.0 / (1.0 + math.exp(-z))

    # -- streaming entry point -----------------------------------------------------------------
    def process(self, samples, return_activations=False):
        x = np.asarray(samples, np.float64).ravel()
        rect = np.empty((NB, len(x)))
        for b in range(NB):
            bb, aa = self.coef[b]
            y, self.zi[b][0] = lfilter(bb, aa, x, zi=self.zi[b][0])
            y, self.zi[b][1] = lfilter(bb, aa, y, zi=self.zi[b][1])
            rect[b] = np.abs(y)
        buf = np.concatenate([self.pending, rect], 1)
        nfr = buf.shape[1] // self.hop
        base = self.nsamples - self.pending.shape[1]         # sample index of buf[:, 0]
        self.pending = buf[:, nfr * self.hop:]
        self.nsamples += len(x)
        events, acts = [], []
        for i in range(nfr):
            pk = buf[:, i * self.hop:(i + 1) * self.hop].max(1)
            E = np.empty(NB)
            for b in range(NB):
                h = self.hold[b]; h[self.hpos[b]] = pk[b]; self.hpos[b] = (self.hpos[b] + 1) % len(h); E[b] = h.max()
            self.g = max(self.g * self.dec, float(E.max()))
            self.r = np.maximum(self.r * self.dec, E)
            F = (1 + np.log10((E + EPS * self.g) / self.g) / 1.5).astype(np.float32)
            C = (1 + np.log10((self.r + EPS * self.g) / self.g) / 1.5).astype(np.float32)
            a = self._frame(F, C)
            self.raw_activation = a
            a = self._tempo(a)
            self.activation = a
            acts.append(a)
            self.since_last += 1
            if a >= self.threshold and self.since_last >= self.refractory:
                self.since_last = 0
                t = (base + (i + 1) * self.hop) / self.sr
                events.append({"time": t, "onset_time": t - self.latency, "strength": a})
        return (events, acts) if return_activations else events


if __name__ == "__main__":
    import sys
    import soundfile as sf
    x, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True)
    det = KickDetector(sr)
    for ev in det.process(x.mean(1)):
        print(f"{ev['time']:.4f}\t{ev['onset_time']:.4f}\t{ev['strength']:.3f}")
