"""Compact plot: network-input band image + low spectrogram with ref (green) / det (cyan/red)."""
import sys, json, numpy as np
sys.path.insert(0, "tools"); import nn as N, frontend as fe, evaluate as ev
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def plot(items, out, detdir="data/det_{}.json"):
    fig, ax = plt.subplots(len(items), 1, figsize=(16, 4.2 * len(items)))
    ax = np.atleast_1d(ax)
    for k, (name, t0, t1) in enumerate(items):
        F, C, fr = N.load_feats(name); a, b = int(t0 * fr), int(t1 * fr)
        ax[k].imshow(F[a:b].T, origin="lower", aspect="auto", extent=[t0, t1, -0.5, 12.5], vmin=-0.6, vmax=1, cmap="magma")
        ax[k].set_yticks(range(13)); ax[k].set_yticklabels(fe.FC)
        for r in json.load(open(f"data/ref/{name}.json")):
            if t0 <= r < t1: ax[k].plot([r, r], [6.5, 12.5], color="lime", lw=1.5)
        for r in json.load(open(detdir.format(name))):
            if t0 <= r < t1: ax[k].plot([r, r], [-0.5, 6], color="cyan", lw=1.2)
        ax[k].set_title(f"{name}   top/green = reference, bottom/cyan = detector")
    plt.tight_layout(); plt.savefig(out, dpi=68); plt.close()
if __name__ == "__main__":
    it = [(sys.argv[i], float(sys.argv[i + 1]), float(sys.argv[i + 2])) for i in range(2, len(sys.argv), 3)]; plot(it, sys.argv[1])
