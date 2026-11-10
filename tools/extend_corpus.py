"""Add N more random tracks (2 excerpts each) to the corpus."""
import os, random, subprocess, json, sys, hashlib
from concurrent.futures import ThreadPoolExecutor
ROOT = "/Volumes/Public/Shared Music"; HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(sys.argv[1]); seed = int(sys.argv[2]); STYLE = sys.argv[3] if len(sys.argv) > 3 else "extra"
files = [l.rstrip("\n") for l in open(os.path.join(HERE, "data/filelist.txt")) if not l.startswith("./Music/")]
cur = json.load(open(os.path.join(HERE, "data/corpus.json"))); have = {c["src"] for c in cur}
rng = random.Random(seed); rng.shuffle(files); pick = [f for f in files if f not in have][:N]
def dur(p):
    try: return float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",p],capture_output=True,text=True,timeout=60).stdout.strip())
    except Exception: return 0.0
def work(f):
    src = os.path.join(ROOT, f[2:]); d = dur(src); out = []
    if d < 100 or d > 1200: return out
    hid = hashlib.md5(f.encode()).hexdigest()[:8]
    for tag, frac in (("a", 0.30), ("b", 0.60)):
        name = f"{STYLE}_{hid}_{tag}"; dst = os.path.join(HERE, "corpus", name + ".flac")
        if not os.path.exists(dst):
            r = subprocess.run(["ffmpeg","-v","error","-y","-ss",f"{d*frac:.2f}","-t","40","-i",src,"-ar","44100","-ac","2","-map","0:a:0",dst],capture_output=True,text=True)
            if r.returncode: continue
        out.append({"id": name, "style": STYLE, "src": f, "start": round(d*frac, 2)})
    return out
with ThreadPoolExecutor(6) as ex: res = [x for r in ex.map(work, pick) for x in r]
json.dump(cur + res, open(os.path.join(HERE, "data/corpus.json"), "w"), indent=1, ensure_ascii=False); print("added", len(res))
