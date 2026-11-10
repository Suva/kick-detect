"""Write docs/RESULTS.md from cross-validated activations (out-of-fold: every excerpt is scored by a model
that never saw its track).  python tools/report.py accurate=data/acts_X.npz:30:THR:THR_TEMPO fast=...:14:THR:THR_TEMPO"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev, nn as N, rhythm
HERE = ev.HERE; fr = 501.136; meta = ev.meta(); BETA = 3.0
STYLE = {"hardtechno": "Hard techno / schranz / hard dance (hand-picked)", "tekno": "Tekno / acid (hand-picked)", "techno": "Techno (hand-picked)",
         "trance": "Trance / psy (hand-picked)", "synthwave": "Synthwave / darkwave (hand-picked)", "electro": "Electro (hand-picked)", "breaks": "Breaks / IDM (hand-picked)"}
def logit(a): a = np.clip(a, 1e-6, 1 - 1e-6); return np.log(a / (1 - a))
def prf(tp, fp, fn): p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); return p, r, 2 * p * r / max(p + r, 1e-9)
def row(label, tp, fp, fn, extra=""):
    p, r, f = prf(tp, fp, fn); return f"| {label} | {tp + fn} | {fn} ({100 * fn / max(tp + fn, 1):.1f}%) | {fp} ({100 * fp / max(tp + fp, 1):.1f}%) | {p:.3f} | {r:.3f} | {f:.3f} |{extra}\n"
def table(per, names):
    out = "| set | kicks | missed (FN) | false (FP) | precision | recall | F1 |\n|---|---|---|---|---|---|---|\n"
    groups = {}
    for n in names:
        st = n.split("_")[0]
        if meta[n].get("empty"): g = "zz"
        elif st in STYLE: g = STYLE[st]
        else:
            b = meta[n].get("bpm", 0); g = "Library random sample, " + ("< 128 BPM" if b < 128 else "128–139 BPM" if b < 140 else "140–149 BPM" if b < 150 else "150–164 BPM" if b < 165 else "≥ 165 BPM")
        groups.setdefault(g, np.zeros(3, int)); groups[g] += per[n]
    tot = np.zeros(3, int)
    for g in sorted(groups):
        if g == "zz": continue
        out += row(g, *groups[g]); tot += groups[g]
    out += row("**All excerpts with kicks**", *tot)
    return out, groups.get("zz", np.zeros(3, int)), tot
def run(spec):
    name, rest = spec.split("="); path, lat, thr, thr_t = rest.split(":"); lat = float(lat); thr = float(thr); thr_t = float(thr_t)
    acts = np.load(path); names = [n for n in ev.names(True) if n in acts]; ev.POST = max(0.07, lat / 1000 + 0.035)
    res = {}
    for tempo in (True, False):
        A = {}
        for n in names:
            a = acts[n]
            if tempo: a = 1 / (1 + np.exp(-(logit(a) + BETA * (2 * rhythm.run(a, fr)[0] - 1))))
            A[n] = (a, fr)
        tot, per, lats = N.score(A, thr_t if tempo else thr, names)
        res[tempo] = (per, lats, names)
    return name, lat, res
if __name__ == "__main__":
    md = ""
    for spec in sys.argv[1:]:
        name, lat, res = run(spec)
        for tempo in (True, False):
            per, lats, names = res[tempo]; t, neg, tot = table(per, names)
            nneg = sum(1 for n in names if meta[n].get("empty")); mins = nneg * 38 / 60
            md += f"\n### Mode `{name}` — tempo assist {'on' if tempo else 'off'}\n\n" + t
            md += f"\nReport latency after the kick onset: median {1e3 * np.median(lats):.0f} ms (10th–90th percentile {1e3 * np.percentile(lats, 10):.0f}–{1e3 * np.percentile(lats, 90):.0f} ms).  \n"
            md += f"Kick-free excerpts (ambient, beatless intros, breakdowns; {nneg} excerpts, {mins:.0f} min): {neg[1]} false triggers = {neg[1] / max(mins, 1e-9):.1f} per minute.  \n"
            errs = sorted(((v[1] + v[2]) for v in per.values()), reverse=True); n0 = sum(1 for e in errs if e == 0); n2 = sum(1 for e in errs if e <= 2)
            md += f"Per excerpt (38 s scored each): {n0} of {len(errs)} excerpts have zero errors, {n2} have at most two; the worst 10 excerpts account for {sum(errs[:10])} of {sum(errs)} errors.\n"
            print(name, "tempo" if tempo else "plain", "ALL", prf(*tot), "neg FP", neg[1])
    open(os.path.join(HERE, "data/results_tables.md"), "w").write(md); print("wrote data/results_tables.md")
