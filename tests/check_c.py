"""Build the C command-line tool for each weight set and compare it with the stored vectors.
   python tests/check_c.py            (set CC / CFLAGS in the environment if needed)"""
import os, sys, subprocess, numpy as np, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); V = os.path.join(HERE, "tests/vectors"); K = os.path.join(HERE, "kickdetect")
meta = json.load(open(os.path.join(V, "clip.json"))); x = open(os.path.join(V, "clip.f32"), "rb").read(); ok = True
for mode in ("accurate", "fast"):
    if not os.path.exists(os.path.join(K, f"kd_weights_{mode}.h")): continue
    exe = os.path.join(HERE, "tests", f"kd_cli_{mode}")
    cmd = [os.environ.get("CC", "cc"), "-O2"] + os.environ.get("CFLAGS", "").split() + [f'-DKD_WEIGHTS_HEADER="kd_weights_{mode}.h"', "-I", K, "-o", exe, os.path.join(K, "kd_cli.c"), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stderr); sys.exit(1)
    for ta in (True, False):
        tag = f"{mode}_{'tempo' if ta else 'notempo'}"; flags = [] if ta else ["--no-tempo"]
        a = np.array(subprocess.run([exe, str(meta["sample_rate"]), "--act"] + flags, input=x, capture_output=True).stdout.decode().split(), float)
        e = np.loadtxt(os.path.join(V, f"activations_{tag}.txt"))
        ev_c = subprocess.run([exe, str(meta["sample_rate"])] + flags, input=x, capture_output=True).stdout.decode()
        ev_e = open(os.path.join(V, f"events_{tag}.txt")).read()
        tc = [float(l.split()[0]) for l in ev_c.strip().split("\n") if l]; te = [float(l.split()[0]) for l in ev_e.strip().split("\n") if l]
        same = len(tc) == len(te) and all(abs(p - q) < 0.0031 for p, q in zip(tc, te))
        d = float(np.abs(a[100:] - e[100:len(a)]).max()); good = d < 2e-3 and same; ok &= good
        print(f"{tag:18s} frames={len(a)} max|C-ref|={d:.2e} events C={len(tc)} ref={len(te)} {'OK' if good else 'MISMATCH'}")
sys.exit(0 if ok else 1)
