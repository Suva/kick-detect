"""Run Demucs on corpus excerpts; save mono drums stem (for reference annotation only)."""
import os, sys, glob, time, numpy as np, soundfile as sf, torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(HERE, "stems"), exist_ok=True)
model = get_model("htdemucs"); model.eval()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
di = model.sources.index("drums"); bi = model.sources.index("bass")
t0 = time.time()
for i, p in enumerate(sorted(glob.glob(os.path.join(HERE, "corpus/*.flac")))):
    name = os.path.basename(p)[:-5]
    out = os.path.join(HERE, "stems", name + "_drums.flac")
    if os.path.exists(out): continue
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    w = torch.from_numpy(x.T.copy())
    ref = w.mean(0); m, s = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        est = apply_model(model, ((w - m) / s)[None], device=dev, shifts=0, split=True, overlap=0.25)[0]
    est = est * s + m
    sf.write(out, est[di].mean(0).cpu().numpy(), sr)
    sf.write(os.path.join(HERE, "stems", name + "_bass.flac"), est[bi].mean(0).cpu().numpy(), sr)
    print(i, name, f"{time.time()-t0:.0f}s", flush=True)
