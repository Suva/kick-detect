"""Train the final model on all trusted excerpts and export it (C header + JSON) with test vectors.
   python tools/final.py --lat 30 --thr 0.95 --name accurate [--ch 16,24,24 --chid 32 --epochs 14]"""
import os, sys, json, argparse, subprocess, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nn as N, evaluate as ev, export
HERE = ev.HERE
ap = argparse.ArgumentParser(); ap.add_argument("--lat", type=int, default=30); ap.add_argument("--thr", type=float, default=0.97); ap.add_argument("--thr-tempo", type=float, default=0.985)
ap.add_argument("--name", default="accurate"); ap.add_argument("--ch", default="12,16,16"); ap.add_argument("--chid", type=int, default=24)
ap.add_argument("--epochs", type=int, default=14); ap.add_argument("--bs", type=int, default=1024); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--semi", action="store_true")
A = ap.parse_args()
fr = 501.136; lat_f = int(round(A.lat / 1000 * fr)); names = ev.names(True)
m = N.train(names, lat_f, N.CNet.RF, epochs=A.epochs, arch="cnn", bs=A.bs, ch=tuple(int(c) for c in A.ch.split(",")), chid=A.chid, seed=A.seed, verbose=True, semi=ev.names(False) if A.semi else ())
torch.save(m.state_dict(), os.path.join(HERE, f"data/model_{A.name}.pt"))
out = os.path.join(HERE, "kickdetect")
meta = dict(name=A.name, latency_ms=round(1000 * lat_f / fr, 1), train_excerpts=len(names), train_tracks=len({n.split('_')[1] for n in names}))
nparam = export.export(m.cpu().eval(), lat_f, A.thr, out_dir=out, meta=meta, threshold_tempo=A.thr_tempo)
os.replace(os.path.join(out, "kd_weights.h"), os.path.join(out, f"kd_weights_{A.name}.h"))
os.replace(os.path.join(out, "weights.json"), os.path.join(out, f"weights_{A.name}.json"))
print("exported", A.name, "params", nparam, meta)
