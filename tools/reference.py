"""Offline (non-causal) reference kick annotator. Uses the Demucs drums stem plus the mix.
Not part of the real-time algorithm: only produces ground truth for evaluation."""
import os, sys, glob, json, numpy as np, soundfile as sf
from scipy.signal import butter, sosfiltfilt, find_peaks, hilbert
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ER = 1000  # envelope rate Hz

def band_env(x, sr, lo, hi, smooth_hz=45.0):
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    y = sosfiltfilt(sos, x)
    n = len(y); N = 1 << (n - 1).bit_length()
    y = np.abs(hilbert(y, N)[:n])
    sos2 = butter(2, smooth_hz, fs=sr, output="sos")
    e = np.maximum(sosfiltfilt(sos2, y), 0)
    step = sr // ER  # 44 -> ~1002 Hz
    return e[::step], sr / step

def annotate(name, P=None, debug=False):
    """Kick = sharp onset in the drums-stem punch band (120-600 Hz) that is followed by a low body (35-120 Hz)."""
    P = dict(punch_rel=0.45, body_rel=0.4, abs_db=-30.0, min_gap=0.050, body_win=0.090, body_gain=1.25) | (P or {})
    d, sr = sf.read(os.path.join(HERE, "stems", name + "_drums.flac"), dtype="float32")
    x, _ = sf.read(os.path.join(HERE, "corpus", name + ".flac"), dtype="float32", always_2d=True)
    x = x.mean(1)
    ep, er = band_env(d, sr, 120, 600, 100.0)     # punch
    el, _ = band_env(d, sr, 35, 120, 45.0)        # body
    full_rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    floor = full_rms * 10 ** (P["abs_db"] / 20)
    dd = np.gradient(ep) * er
    pk, _ = find_peaks(dd, distance=int(P["min_gap"] * er))
    w = int(0.020 * er); bw = int(P["body_win"] * er)
    c = []
    for p in pk:
        a = max(0, p - w); b = min(len(ep), p + w + 1)
        prise = ep[p:b].max() - ep[a:p + 1].min()
        body = el[p:min(len(el), p + bw)].max()
        pre = el[max(0, p - w):p + w].min()
        c.append((p / er, prise, body, pre))
    if not c: return []
    c = np.array(c)
    big = c[c[:, 1] >= floor]
    if len(big) < 3: return []
    Kp = np.percentile(big[:, 1], 95)                 # kicks are among the largest punch-band drum onsets
    strong = big[big[:, 1] >= 0.5 * Kp]
    Kb = np.percentile(strong[:, 2], 80)              # typical body level after strong punches
    out = []
    for t, prise, body, pre in c:
        if prise < max(P["punch_rel"] * Kp, floor): continue
        if body < max(P["body_rel"] * Kb, floor): continue
        if body < P["body_gain"] * pre: continue        # body must rise after the punch (not just ongoing bass/rumble)
        out.append(round(float(t), 4))
    return (out, c, Kp, Kb) if debug else out

def annotate_tm(name, P=None, debug=False):
    """v4: per-track kick template matching on the drums-stem low spectrogram, seeded by annotate()."""
    from scipy.signal import spectrogram
    P = dict(rho=0.55, alpha=0.40, seed_frac=0.3, pre=0.015, post=0.110, cand_rel=0.15, min_gap=0.055) | (P or {})
    r0 = annotate(name, debug=True)
    if not isinstance(r0, tuple): return []
    seeds, c, Kp, Kb = r0
    if len(seeds) < 6: return list(seeds)
    d, sr = sf.read(os.path.join(HERE, "stems", name + "_drums.flac"), dtype="float32")
    hop = 64
    f, t, S = spectrogram(d, sr, nperseg=1024, noverlap=1024 - hop, mode="magnitude")
    S = S[1:24]                                   # 43..990 Hz
    S = np.log1p(S / (np.percentile(S, 99) * 0.02 + 1e-9))
    fr = sr / hop; off = 512 / sr                 # frame i is centred at t[i]
    a = int(P["pre"] * fr); b = int(P["post"] * fr)
    def patch(tt):
        i = int(round((tt - off) * fr))
        if i - a < 0 or i + b > S.shape[1]: return None
        return S[:, i - a:i + b]
    ct = c[:, 0]; pr = c[:, 1]
    seedset = set(seeds); sidx = [i for i in range(len(ct)) if round(float(ct[i]), 4) in seedset]
    sidx = sorted(sidx, key=lambda i: -pr[i])[: max(6, int(len(sidx) * P["seed_frac"]))]
    cidx = [i for i in range(len(ct)) if pr[i] >= P["cand_rel"] * Kp and patch(ct[i]) is not None]
    Pm = {i: patch(ct[i]) for i in cidx}
    use = [i for i in sidx if i in Pm]
    if len(use) < 4: return list(seeds)
    acc = use
    for it in range(3):
        T = np.median([Pm[i] for i in acc], 0)
        Tz = T - T.mean(); tn = np.sqrt((Tz ** 2).sum()) + 1e-9
        sc = {}
        for i in cidx:
            X = Pm[i]; Xz = X - X.mean()
            rho = float((Xz * Tz).sum() / (tn * (np.sqrt((Xz ** 2).sum()) + 1e-9)))
            alpha = float((X * T).sum() / ((T * T).sum() + 1e-9))
            sc[i] = (rho, alpha)
        acc = [i for i in cidx if sc[i][0] >= P["rho"] and sc[i][1] >= P["alpha"]]
        if len(acc) < 4: return list(seeds)
    acc.sort(key=lambda i: ct[i]); out = []
    for i in acc:
        if out and ct[i] - ct[out[-1]] < P["min_gap"]:
            if sc[i][0] * sc[i][1] > sc[out[-1]][0] * sc[out[-1]][1]: out[-1] = i
        else: out.append(i)
    res = [round(float(ct[i]), 4) for i in out]
    return (res, {round(float(ct[i]), 4): sc[i] for i in cidx}) if debug else res

def tempo_grid(o, er):
    """Find beat period/phase from onset-strength o (rate er). Returns (T, phase, beat_times) or None."""
    n = len(o); k = int(0.008 * er); os_ = np.convolve(o, np.ones(k) / k, "same")
    x = os_ - os_.mean(); N = 1 << (2 * n - 1).bit_length()
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, N)) ** 2)[:n]
    lo = int(60 / 215 * er); hi = int(60 / 105 * er)
    L0 = lo + int(np.argmax(ac[lo:hi]))
    tt = np.arange(n) / er
    def fold(T):
        nb = 200; ph = ((tt % T) / T * nb).astype(int) % nb
        acc = np.bincount(ph, os_, nb) / np.maximum(np.bincount(ph, None, nb), 1)
        j = int(np.argmax(acc)); return acc[j], (j + 0.5) / nb * T
    best = (0, None, None)
    for T in np.arange(0.985, 1.0151, 0.00005) * (L0 / er):
        sc, ph = fold(T)
        if sc > best[0]: best = (sc, T, ph)
    sc, T, ph = best
    if T / 2 >= 60 / 215:
        sc2, ph2 = fold(T / 2)
        if sc2 >= 0.7 * sc: T, ph, sc = T / 2, ph2, sc2
    return T, ph, np.arange(ph, n / er, T), os_

def annotate_grid(name, P=None, debug=False, stem="drums"):
    """v5: tempo-grid-seeded template matching. On-grid hits are accepted leniently, off-grid hits strictly."""
    from scipy.signal import spectrogram
    P = dict(rho_on=0.40, alpha_on=0.35, rho_off=0.70, alpha_off=0.50, pre=0.015, post=0.110, min_gap=0.055,
             grid_tol=0.016, abs_db=-30.0) | (P or {})
    d, sr = sf.read(os.path.join(HERE, "stems", name + "_drums.flac"), dtype="float32")
    if stem == "db": d = d + sf.read(os.path.join(HERE, "stems", name + "_bass.flac"), dtype="float32")[0]
    x, _ = sf.read(os.path.join(HERE, "corpus", name + ".flac"), dtype="float32", always_2d=True); x = x.mean(1)
    full_rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9); floor = max(full_rms * 10 ** (P["abs_db"] / 20), 0.02)
    ep, er = band_env(d, sr, 120, 600, 100.0); el, _ = band_env(d, sr, 35, 120, 45.0)
    o = np.maximum(np.gradient(ep) * er, 0)
    pk, _ = find_peaks(o, distance=int(0.012 * er))
    w = int(0.020 * er); bw = int(0.090 * er)
    ct = pk / er
    prise = np.array([ep[p:p + w + 1].max() - ep[max(0, p - w):p + 1].min() for p in pk])
    body = np.array([el[p:p + bw].max() for p in pk]); pre = np.array([el[max(0, p - w):p + w].min() for p in pk])
    if len(pk) < 8 or prise.max() < floor: return ([], {}) if debug else []
    T, ph, beats, os_ = tempo_grid(o, er)
    # nearest candidate to each beat
    gi = []
    for bt in beats:
        j = np.flatnonzero(np.abs(ct - bt) <= P["grid_tol"])
        if len(j): gi.append(j[np.argmax(prise[j])])
    gi = np.array(sorted(set(gi)), int)
    gridded = False
    if len(gi) >= 8:
        z = prise[gi]; z80 = np.percentile(z, 80)
        ok = (z >= 0.4 * z80) & (z >= floor) & (body[gi] >= 1.25 * pre[gi]) & (body[gi] >= 0.4 * np.percentile(body[gi], 80))
        # 4-on-the-floor if most beats (over the busiest half of the excerpt) carry a kick-like hit
        gridded = ok.sum() >= 0.45 * len(beats) and ok.sum() >= 16
    okset = set()
    if gridded:
        seeds = gi[ok & (z >= 0.6 * z80)]; okset = set(int(i) for i in gi[ok])
    else:
        v3 = annotate(name, debug=True) if stem == "drums" else []
        if not isinstance(v3, tuple) or len(v3[0]) < 6: return ([], {}) if debug else (list(v3[0]) if isinstance(v3, tuple) else [])
        st = np.array(v3[0]); seeds = np.array([int(np.argmin(np.abs(ct - t))) for t in st], int)
        seeds = seeds[np.argsort(-prise[seeds])][: max(6, int(0.35 * len(seeds)))]
    hop = 64
    f, t, S = spectrogram(d, sr, nperseg=1024, noverlap=1024 - hop, mode="magnitude")
    S = S[1:24]; S = np.log1p(S / (np.percentile(S, 99) * 0.02 + 1e-9)); S = S - np.median(S, 1, keepdims=True)
    fr = sr / hop; off = 512 / sr; a = int(P["pre"] * fr); b = int(P["post"] * fr)
    def patch(tt):
        i = int(round((tt - off) * fr))
        return None if (i - a < 0 or i + b > S.shape[1]) else S[:, i - a:i + b]
    Kp = np.median(prise[seeds])
    cidx = [i for i in range(len(ct)) if prise[i] >= max(0.15 * Kp, floor) and patch(ct[i]) is not None]
    Pm = {i: patch(ct[i]) for i in cidx}
    acc = [i for i in seeds if i in Pm]
    if len(acc) < 4: return ([], {}) if debug else []
    ongrid = np.zeros(len(ct), bool)
    if gridded:
        for i in cidx: ongrid[i] = np.min(np.abs(beats - ct[i])) <= P["grid_tol"]
    sc = {}
    for it in range(2):
        Tm = np.median([Pm[i] for i in acc], 0); Tz = Tm - Tm.mean(); tn = np.sqrt((Tz ** 2).sum()) + 1e-9
        for i in cidx:
            X = Pm[i]; Xz = X - X.mean()
            sc[i] = (float((Xz * Tz).sum() / (tn * (np.sqrt((Xz ** 2).sum()) + 1e-9))), float((X * Tm).sum() / ((Tm * Tm).sum() + 1e-9)))
        new = [i for i in cidx if (i in okset) or ((sc[i][0] >= P["rho_on"] and sc[i][1] >= P["alpha_on"] and prise[i] >= 0.3 * Kp) if ongrid[i]
                                                   else (sc[i][0] >= P["rho_off"] and sc[i][1] >= P["alpha_off"] and prise[i] >= 0.5 * Kp))]
        if len(new) < 4: break
        acc = new if it == 0 else acc
        final = new
    else:
        pass
    final = locals().get("final", acc)
    final = sorted(final, key=lambda i: ct[i]); out = []
    for i in final:
        if out and ct[i] - ct[out[-1]] < P["min_gap"]:
            if sc[i][0] * sc[i][1] > sc[out[-1]][0] * sc[out[-1]][1]: out[-1] = i
        else: out.append(i)
    res = [round(float(ct[i]), 4) for i in out]
    info = dict(bpm=round(float(60 / T), 2), gridded=bool(gridded), n=len(res), ongrid=int(sum(ongrid[i] for i in out)), beats=len(beats),
                rho=round(float(np.median([sc[i][0] for i in out])), 3) if out else 0.0,
                rho10=round(float(np.percentile([sc[i][0] for i in out], 10)), 3) if out else 0.0)
    return (res, info) if debug else res

def punch_cands(name):
    """All drum/bass-stem punch-band onsets of notable size (kicks, snares, toms, bass attacks...).
    Frames far from every such onset are 'safe negatives' for training even when kick labels are incomplete."""
    d, sr = sf.read(os.path.join(HERE, "stems", name + "_drums.flac"), dtype="float32")
    d = d + sf.read(os.path.join(HERE, "stems", name + "_bass.flac"), dtype="float32")[0]
    ep, er = band_env(d, sr, 120, 600, 100.0)
    o = np.maximum(np.gradient(ep) * er, 0); pk, _ = find_peaks(o, distance=int(0.012 * er)); w = int(0.020 * er)
    if len(pk) == 0: return []
    prise = np.array([ep[p:p + w + 1].max() - ep[max(0, p - w):p + 1].min() for p in pk])
    big = prise[prise >= 0.02]
    if len(big) < 3: return [round(float(p / er), 4) for p, r in zip(pk, prise) if r >= 0.02]
    Kp = np.percentile(big, 95)
    return [round(float(p / er), 4) for p, r in zip(pk, prise) if r >= max(0.2 * Kp, 0.01)]

def annotate_best(name):
    """Pick drums-stem annotation unless only drums+bass yields a consistent 4/4 grid (kick leaked into bass stem)."""
    k, info = annotate_grid(name, debug=True, stem="drums")
    def cons(i): return bool(i) and i.get("gridded") and i["ongrid"] >= 0.5 * i["beats"] and (i["n"] - i["ongrid"]) <= 0.7 * i["ongrid"]
    info = dict(info); info["stem"] = "drums"
    if not cons(info):
        k2, i2 = annotate_grid(name, debug=True, stem="db")
        if cons(i2) and i2["n"] > 1.5 * len(k): k, info = k2, dict(i2, stem="db")
    # trust: consistent grid, or (near-)empty with nothing on either stem
    info["trusted"] = bool(cons(info)) and info.get("rho", 0) >= 0.8 and info.get("rho10", 0) >= 0.3
    if len(k) == 0:
        k2, i2 = annotate_grid(name, debug=True, stem="db")
        info["trusted"] = len(k2) <= 2; info["empty"] = True
    return k, info

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "data/ref"), exist_ok=True); os.makedirs(os.path.join(HERE, "data/ref_cand"), exist_ok=True)
    meta = {}
    for p in sorted(glob.glob(os.path.join(HERE, "stems/*_drums.flac"))):
        name = os.path.basename(p)[:-11]
        if len(sys.argv) > 1 and not any(a in name for a in sys.argv[1:]): continue
        k, info = annotate_best(name)
        json.dump(k, open(os.path.join(HERE, "data/ref", name + ".json"), "w")); meta[name] = info
        json.dump(punch_cands(name), open(os.path.join(HERE, "data/ref_cand", name + ".json"), "w"))
        print(f"{name:28s} n={len(k):4d} {info}")
    if len(sys.argv) == 1: json.dump(meta, open(os.path.join(HERE, "data/ref_meta.json"), "w"), indent=1)
