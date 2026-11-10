"""Pick a style-diverse set of tracks from the library and cut local excerpts."""
import os, random, re, subprocess, json, sys, hashlib
from concurrent.futures import ThreadPoolExecutor
ROOT = "/Volumes/Public/Shared Music"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = [l.rstrip("\n") for l in open(os.path.join(HERE, "data/filelist.txt"))]
files = [f for f in files if not f.startswith("./Music/")]  # duplicate tree
rng = random.Random(7)
CATS = [  # (style, regex, count)
    ("hardtechno", r"hydraulix", 8), ("hardtechno", r"schranz", 6),
    ("hardtechno", r"sandy warez|hard drums|endymion|uptempo|hardcore", 4),
    ("tekno", r"acidpach|tribe|tekno", 4),
    ("techno", r"dax[ _]j", 4), ("techno", r"ancient methods", 4),
    ("techno", r"arnaud le texier|a[._]paul|altinbas", 5),
    ("electro", r"dynarec|cosmic force|electro u\.n\.i\.t\.y|jensen interceptor|electro guilde", 7),
    ("trance", r"^\./Trance/|darude.*\.mp3$", 6),
    ("breaks", r"bricolage|/ASC ", 6),
    ("synthwave", r"corvad|strngr|hante\.", 5),
    ("nokick", r"musick to play|bvdub|underwater ambience", 5),
]
chosen = {}
for style, pat, n in CATS:
    c = [f for f in files if re.search(pat, f, re.I) and f not in chosen]
    rng.shuffle(c)
    for f in c[:n]: chosen[f] = style
rest = [f for f in files if f not in chosen]
rng.shuffle(rest)
for f in rest[:30]: chosen[f] = "random"

def dur(path):
    try:
        o = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",path],capture_output=True,text=True,timeout=60).stdout
        return float(o.strip())
    except Exception: return 0.0

def work(item):
    f, style = item
    src = os.path.join(ROOT, f[2:])
    d = dur(src)
    if d < 100: return []
    out = []
    hid = hashlib.md5(f.encode()).hexdigest()[:8]
    for tag, frac in (("a", 0.30), ("b", 0.60)):
        name = f"{style}_{hid}_{tag}"
        dst = os.path.join(HERE, "corpus", name + ".flac")
        if not os.path.exists(dst):
            r = subprocess.run(["ffmpeg","-v","error","-y","-ss",f"{d*frac:.2f}","-t","40","-i",src,"-ar","44100","-ac","2","-map","0:a:0",dst],capture_output=True,text=True)
            if r.returncode != 0: continue
        out.append({"id": name, "style": style, "src": f, "start": round(d*frac,2)})
    return out
with ThreadPoolExecutor(6) as ex:
    res = [x for r in ex.map(work, chosen.items()) for x in r]
json.dump(res, open(os.path.join(HERE, "data/corpus.json"), "w"), indent=1, ensure_ascii=False)
print(len(chosen), "tracks ->", len(res), "excerpts")
