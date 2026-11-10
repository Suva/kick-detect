"""Tiny causal network on a sliding window of log band levels. Group-CV by track + final training/export."""
import sys, os, json, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev, frontend as fe
HERE = ev.HERE; WARM = 2.0; REFR = 0.060; NB = fe.NB
dev = "mps" if torch.backends.mps.is_available() else "cpu"
os.makedirs(os.path.join(HERE, "data/feat"), exist_ok=True)

def load_feats(name):
    p = os.path.join(HERE, "data/feat", name + ".npz")
    if not os.path.exists(p):
        x, sr = ev.load(name); F, C, fr = fe.features(x, sr); np.savez(p, F=F, C=C, fr=fr)
    z = np.load(p); return z["F"], z["C"], float(z["fr"])

class Net(nn.Module):
    def __init__(s, T, hid=(32, 16), drop=0.1):
        super().__init__(); d = 2 * NB * T + NB; L = []
        for h in hid: L += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]; d = h
        L += [nn.Linear(d, 1)]; s.f = nn.Sequential(*L); s.T = T
    def forward(s, x): return s.f(x).squeeze(-1)

class CNet(nn.Module):
    """Causal dilated conv net over (time x band). Receptive field 29 frames; output uses only the newest column."""
    RF = 29
    def __init__(s, T=29, ch=(12, 16, 16), hid=24, drop=0.1):
        super().__init__(); s.T = T
        s.c1 = nn.Conv2d(2, ch[0], (5, 3)); s.c2 = nn.Conv2d(ch[0], ch[1], (5, 3), dilation=(2, 1)); s.c3 = nn.Conv2d(ch[1], ch[2], (5, 3), dilation=(4, 1))
        s.d1 = nn.Linear(ch[2] * (NB - 6) + NB, hid); s.d2 = nn.Linear(hid, 1); s.drop = nn.Dropout(drop)
    def forward(s, x):
        B = x.shape[0]; T = s.T; w = x[:, :2 * NB * T].reshape(B, 2, T, NB); c = x[:, 2 * NB * T:]
        h = torch.relu(s.c1(w)); h = torch.relu(s.c2(h)); h = torch.relu(s.c3(h))        # (B, C, 1, NB-6)
        h = s.drop(h[:, :, -1, :].reshape(B, -1))
        return s.d2(torch.relu(s.d1(torch.cat([h, c], 1)))).squeeze(-1)

def build(names, lat_f, T, semi=()):
    """Concatenate excerpts; return feature tensors and labels (1 pos, 0 neg, -1 ignore).
    `semi` excerpts have incomplete kick labels: only frames far from any drum/bass onset count as negatives,
    and their reference kicks count as positives only when the annotation is self-consistent (rho >= 0.8)."""
    Fs, Cs, ys, off = [], [], [], 0; spans = {}; meta = ev.meta()
    for name in list(names) + list(semi):
        F, C, fr = load_feats(name); n = len(F)
        ref = json.load(open(os.path.join(HERE, "data/ref", name + ".json")))
        if name in semi:
            y = np.zeros(n, np.int8); usepos = meta.get(name, {}).get("rho", 0) >= 0.8
            cp = os.path.join(HERE, "data/ref_cand", name + ".json")
            for c in (json.load(open(cp)) if os.path.exists(cp) else []) + ref:
                a = int((c - 0.060) * fr); b = int((c + 0.060) * fr) + lat_f; y[max(a, 0):min(b, n)] = -1
            if not os.path.exists(cp): y[:] = -1
            if not usepos: ref = []
        else:
            y = np.zeros(n, np.int8)
            for r in ref:
                a = int((r - 0.020) * fr); b = int(r * fr) + lat_f + 20; y[max(a, 0):min(b, n)] = -1
        for r in ref:
            c = int(round(r * fr)) + lat_f; y[max(c - 1, 0):min(c + 2, n)] = 1
        y[: int(WARM * fr)] = -1
        Fs.append(F); Cs.append(C); ys.append(y); spans[name] = (off, n, fr, ref); off += n
    return np.concatenate(Fs), np.concatenate(Cs), np.concatenate(ys), spans

def windows(Ft, Ct, idx, T, tilt=0.0):
    ii = (idx[:, None] - torch.arange(T - 1, -1, -1, device=idx.device)[None]).clamp_min(0)
    w = Ft[ii]; c = Ct[idx]; d = w - Ct[ii]          # (B,T,NB): level rel. global peak, and rel. each band's own slow peak (per frame)
    if tilt > 0:   # random smooth EQ offset per sample (same for window and context)
        B = len(idx); a = torch.randn(B, 1, device=idx.device) * tilt; s = torch.randn(B, 1, device=idx.device) * tilt
        e = a + s * torch.linspace(-1, 1, NB, device=idx.device)[None] + torch.randn(B, NB, device=idx.device) * tilt * 0.5
        w = w + e[:, None, :]; c = c + e
    return torch.cat([w.reshape(len(idx), -1), d.reshape(len(idx), -1), c], 1)

def train(names, lat_f, T, epochs=14, seed=0, hid=(32, 16), verbose=False, arch="mlp", bs=2048, ch=(12, 16, 16), chid=24, semi=()):
    torch.manual_seed(seed); np.random.seed(seed)
    F, C, y, _ = build(names, lat_f, T, semi)
    Ft = torch.tensor(F, device=dev); Ct = torch.tensor(C, device=dev)
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    m = (CNet(T, ch=ch, hid=chid) if arch == "cnn" else Net(T, hid)).to(dev); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=epochs * 400)
    for ep in range(epochs):
        m.train(); tot = 0
        for it in range(400):
            npos = bs // 4; ip = np.random.choice(pos, npos); ineg = np.random.choice(neg, bs - npos)
            idx = torch.tensor(np.concatenate([ip, ineg]), device=dev)
            X = windows(Ft, Ct, idx, T, tilt=0.06); Y = torch.cat([torch.ones(npos), torch.zeros(bs - npos)]).to(dev)
            loss = nn.functional.binary_cross_entropy_with_logits(m(X), Y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if verbose: print(f"  ep{ep} loss {tot/400:.4f}", flush=True)
    return m

@torch.no_grad()
def activations(m, name, T):
    F, C, fr = load_feats(name); Ft = torch.tensor(F, device=dev); Ct = torch.tensor(C, device=dev); m.eval()
    out = [torch.sigmoid(m(windows(Ft, Ct, torch.arange(a, min(a + 4096, len(F)), device=dev), T))) for a in range(0, len(F), 4096)]
    return torch.cat(out).cpu().numpy(), fr

def pick(act, fr, thr, refr=REFR):
    det = []; last = -1e9
    for i in np.flatnonzero(act >= thr):
        t = i / fr
        if t - last >= refr: det.append(t); last = t
    return det

def score(acts, thr, names):
    tot = {}; per = {}; lat = []
    for name in names:
        a, fr = acts[name]; ref = [r for r in json.load(open(os.path.join(HERE, "data/ref", name + ".json"))) if r >= WARM]
        det = [t for t in pick(a, fr, thr) if t >= WARM]
        tp, fp, fn, pairs = ev.match(ref, det); per[name] = (tp, fp, fn); lat += [d - r for r, d in pairs]
        for s in (name.split("_")[0], "ALL"): tot[s] = tot.get(s, np.zeros(3, int)) + (tp, fp, fn)
    return tot, per, lat

def cv(lat_ms=16, ctx_ms=32, folds=5, hid=(32, 16), epochs=14, tag="", arch="mlp", bs=2048, ch=(12, 16, 16), chid=24, semi=False):
    names_t = ev.names(True); names_u = ev.names(False)
    fr = 501.1; lat_f = int(round(lat_ms / 1000 * fr)); T = CNet.RF if arch == "cnn" else lat_f + int(round(ctx_ms / 1000 * fr))
    tracks = sorted({n.split("_")[1] for n in names_t + names_u}); np.random.default_rng(1).shuffle(tracks); fold = {t: i % folds for i, t in enumerate(tracks)}
    acts = {}; t0 = time.time()
    for k in range(folds):
        m = train([n for n in names_t if fold[n.split("_")[1]] != k], lat_f, T, epochs=epochs, hid=hid, arch=arch, bs=bs, ch=ch, chid=chid,
                  semi=[n for n in names_u if fold[n.split("_")[1]] != k] if semi else ()); print(f"  fold {k} trained {time.time()-t0:.0f}s", flush=True)
        for n in names_t + names_u:
            if fold[n.split("_")[1]] == k: acts[n] = activations(m, n, T)
    best = None
    ev.POST = max(0.070, lat_ms / 1000 + 0.035)
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.985, 0.993):
        tot, per, lat = score(acts, thr, names_t); tp, fp, fn = tot["ALL"]; f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if best is None or f1 > best[0]: best = (f1, thr, tot, per, lat)
    f1, thr, tot, per, lat = best
    print(f"== {arch} lat={lat_ms}ms ctx={ctx_ms}ms T={T} hid={hid} thr={thr}  TRUSTED F={f1:.4f}  ({time.time()-t0:.0f}s)")
    for s, (tp, fp, fn) in sorted(tot.items()): print(f"   {s:11s} TP={tp:5d} FP={fp:4d} FN={fn:4d} P={tp/max(tp+fp,1):.3f} R={tp/max(tp+fn,1):.3f}")
    print(f"   latency ms median={1e3*np.median(lat):.1f} p10={1e3*np.percentile(lat,10):.1f} p90={1e3*np.percentile(lat,90):.1f}")
    w = sorted(per.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))[:14]
    print("   worst:", " ".join(f"{n.split('_')[0][:4]}_{n.split('_',1)[1]}({v[1]},{v[2]})" for n, v in w))
    totu, peru, _ = score(acts, thr, names_u); tp, fp, fn = totu.get("ALL", (0, 0, 0))
    print(f"   UNTRUSTED-ref set: TP={tp} FP={fp} FN={fn} P={tp/max(tp+fp,1):.3f} R={tp/max(tp+fn,1):.3f}")
    np.savez_compressed(os.path.join(HERE, f"data/acts_{arch}_{lat_ms}{tag}.npz"), **{n: a.astype(np.float32) for n, (a, f) in acts.items()})
    json.dump({n: pick(a, f, thr) for n, (a, f) in acts.items()}, open(os.path.join(HERE, f"data/cvdet{tag}.json"), "w"))
    for n, (a, f) in acts.items(): json.dump(pick(a, f, thr), open(os.path.join(HERE, f"data/det_{n}.json"), "w"))
    return best

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("arch"); ap.add_argument("lats", nargs="+"); ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--bs", type=int, default=1024); ap.add_argument("--tag", default=""); ap.add_argument("--ch", default="12,16,16"); ap.add_argument("--chid", type=int, default=24); ap.add_argument("--semi", action="store_true")
    A = ap.parse_args()
    for a in A.lats:
        lat, ctx = a.split(":"); cv(int(lat), int(ctx), arch=A.arch, folds=A.folds, epochs=A.epochs, bs=A.bs, tag=A.tag, ch=tuple(int(c) for c in A.ch.split(",")), chid=A.chid, semi=A.semi)
