"""Diagnostic plot: spectrograms + low-band envelopes with reference/detected marks."""
import os, sys, json, numpy as np, soundfile as sf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import spectrogram
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import band_env, HERE
def plot(name, t0, t1, out, det=None, ref=None):
    x, sr = sf.read(os.path.join(HERE, "corpus", name + ".flac"), dtype="float32", always_2d=True); x = x.mean(1)
    dp = os.path.join(HERE, "stems", name + "_drums.flac")
    d = sf.read(dp, dtype="float32")[0] if os.path.exists(dp) else None
    if ref is None:
        rp = os.path.join(HERE, "data/ref", name + ".json"); ref = json.load(open(rp)) if os.path.exists(rp) else []
    a, b = int(t0 * sr), int(t1 * sr)
    fig, ax = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    f, t, S = spectrogram(x[a:b], sr, nperseg=2048, noverlap=2048 - 128)
    m = f < 400; ax[0].pcolormesh(t + t0, f[m], 10 * np.log10(S[m] + 1e-10), vmin=-80, vmax=-20, shading="auto"); ax[0].set_ylabel("mix <400Hz")
    f, t, S = spectrogram(x[a:b], sr, nperseg=512, noverlap=512 - 128)
    ax[1].pcolormesh(t + t0, f, 10 * np.log10(S + 1e-10), vmin=-100, vmax=-30, shading="auto"); ax[1].set_yscale("symlog", linthresh=200); ax[1].set_ylim(40, 20000); ax[1].set_ylabel("mix full")
    em, er = band_env(x, sr, 30, 150); tt = np.arange(len(em)) / er; s = (tt >= t0) & (tt < t1)
    ax[2].plot(tt[s], em[s], label="mix 30-150"); 
    if d is not None:
        ed, _ = band_env(d, sr, 30, 150); ax[2].plot(tt[s], ed[s], label="drums stem 30-150"); eq, _ = band_env(d, sr, 120, 600, 100.0); ax[2].plot(tt[s], eq[s], label="drums stem 120-600", lw=0.7)
    ax[2].legend(loc="upper right")
    sos_t = np.arange(a, b) / sr
    from scipy.signal import butter, sosfiltfilt
    lp = sosfiltfilt(butter(4, 200, fs=sr, output="sos"), x)
    ax[3].plot(sos_t, lp[a:b], lw=0.5); ax[3].set_ylabel("mix LP200 wave")
    for k in ref:
        if t0 <= k < t1:
            for q in ax: q.axvline(k, color="lime", lw=1.2, ls="--")
    for k in det or []:
        if t0 <= k < t1:
            for q in ax: q.axvline(k, color="red", lw=0.8)
    ax[0].set_title(f"{name}  green dashed=reference  red=detector"); plt.tight_layout(); plt.savefig(out, dpi=70); plt.close()
if __name__ == "__main__":
    plot(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
