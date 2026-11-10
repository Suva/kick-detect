"""Causal front-end shared by training and the reference implementation (numpy version).
Band filterbank (cascaded RBJ band-pass biquads) -> per-hop peak envelopes with short hold -> log, level-normalised."""
import numpy as np
from scipy.signal import lfilter
FC = [45, 64, 90, 128, 180, 256, 360, 512, 720, 1024, 2048, 4096, 8192]
QS = [2.0] * 9 + [1.0] * 4
NB = len(FC); HOP_HZ = 500.0; TAU = 4.0; EPS = 1e-3

def bpf(fc, q, sr):
    w0 = 2 * np.pi * fc / sr; al = np.sin(w0) / (2 * q); a0 = 1 + al
    return np.array([al, 0.0, -al]) / a0, np.array([1.0, -2 * np.cos(w0) / a0, (1 - al) / a0])

def envelopes(x, sr):
    hop = int(round(sr / HOP_HZ)); n = len(x) // hop; fr = sr / hop
    E = np.zeros((NB, n), np.float32)
    for b in range(NB):
        bb, aa = bpf(FC[b], QS[b], sr)
        y = np.abs(lfilter(bb, aa, lfilter(bb, aa, x)))[: n * hop].reshape(n, hop).max(1)
        H = int(np.ceil(0.5 / (FC[b] * hop / sr)))
        out = y.copy()
        for k in range(1, H): out[k:] = np.maximum(out[k:], y[:-k])
        E[b] = out
    return E, fr

def features(x, sr):
    """Returns F (n, NB) log band levels in ~[-1,1] and C (n, NB) slow band references, both relative to global level."""
    E, fr = envelopes(x, sr); n = E.shape[1]
    dec = np.exp(-1.0 / (TAU * fr)); G = np.zeros(n, np.float32); R = np.zeros_like(E)
    g = 1e-4; r = np.full(NB, 1e-4, np.float32); Em = E.max(0)
    for i in range(n):
        g = max(g * dec, Em[i]); r = np.maximum(r * dec, E[:, i]); G[i] = g; R[:, i] = r
    F = 1 + np.log10((E + EPS * G) / G) / 1.5
    C = 1 + np.log10((R + EPS * G) / G) / 1.5
    return F.T.astype(np.float32).copy(), C.T.astype(np.float32).copy(), fr
