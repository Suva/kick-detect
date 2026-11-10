# Results

All numbers below are for the shipped configuration (10,173-weight network, `accurate` = 30 ms nominal
latency, `fast` = 14 ms nominal latency). Nothing here was measured on audio the scored model was trained on.

## Corpus

| | |
|---|---|
| Source | DJ library on the network share (`Shared Music`), ~11,500 audio files |
| Tracks sampled | 508 (94 hand-picked by style: hard techno / schranz, tekno, techno, electro, trance, breaks, synthwave, ambient; 414 drawn at random) |
| Excerpts | 1,016 (two 40 s excerpts per track, starting at 30 % and 60 % of the track) |
| Excerpts with trusted annotation | 578 from 345 tracks: 543 with kicks (44,846 kicks) + 35 kick-free |
| Scored span | 2 s – 40 s of each excerpt (the first 2 s are warm-up for the level followers) |
| Match window | a detection counts if its report time lies within −30 … +70 ms of a reference kick; each reference kick can be matched once |

## How the ground truth was made

No hand labels exist for this library, so the reference is produced offline by a separate, non-causal
pipeline that shares no code or features with the detector (`tools/reference.py`):

1. Demucs (htdemucs) separates each excerpt; the drum stem (or drums + bass when the kick leaked into
   the bass stem) is analysed.
2. Candidate hits are sharp onsets of the 120–600 Hz "punch" band of the stem.
3. An offline tempo/phase estimate gives the beat grid; on-grid hits with a low body seed a per-track
   kick template (kicks in electronic music are near-identical repeated samples).
4. Hits are accepted by template correlation — leniently on the grid, strictly off it.
5. An annotation is **trusted** only if the grid is consistent and the accepted kicks correlate strongly
   with the template (median ρ ≥ 0.8). 438 excerpts fail that test (breakbeats, electro, sparse or
   irregular kicks, heavily layered material) and are not used for the numbers below.

Consequences worth knowing:

- The trusted set is dominated by steady 4/4 material (techno in all its forms, trance, synthwave).
  Electro, breaks and other syncopated styles are under-represented in the *measurement*, not excluded
  from the *detector* — see "Known limitations".
- The reference is conservative about off-grid hits (grace kicks, ghost kicks, kick rolls). Where a
  track has those, real kicks found by the detector are counted as false positives. Inspecting the
  worst excerpts showed this repeatedly, so the false-positive figures are an upper bound.
- 8 excerpts were removed after visual inspection of spectrograms showed the annotation itself
  was wrong (listed with reasons in `data/ref_overrides.json`). No excerpt was removed for any other reason.

## Cross-validation by track (main result)

Three folds, split by track: every excerpt is scored by a model that never saw either excerpt of that
track. Thresholds: `accurate` 0.97 / 0.985 with tempo assist, `fast` 0.97 / 0.993.

"missed" is the share of reference kicks without a detection; "false" is the share of detections
without a reference kick.

### Mode `accurate` — tempo assist on

| set | kicks | missed (FN) | false (FP) | precision | recall | F1 |
|---|---|---|---|---|---|---|
| Hard techno / schranz / hard dance (hand-picked) | 2019 | 21 (1.0%) | 39 (1.9%) | 0.981 | 0.990 | 0.985 |
| Library random sample, 128–139 BPM | 16931 | 395 (2.3%) | 581 (3.4%) | 0.966 | 0.977 | 0.971 |
| Library random sample, 140–149 BPM | 9583 | 144 (1.5%) | 355 (3.6%) | 0.964 | 0.985 | 0.974 |
| Library random sample, 150–164 BPM | 3651 | 107 (2.9%) | 265 (7.0%) | 0.930 | 0.971 | 0.950 |
| Library random sample, < 128 BPM | 6821 | 353 (5.2%) | 293 (4.3%) | 0.957 | 0.948 | 0.952 |
| Library random sample, ≥ 165 BPM | 713 | 62 (8.7%) | 71 (9.8%) | 0.902 | 0.913 | 0.907 |
| Synthwave / darkwave (hand-picked) | 280 | 6 (2.1%) | 13 (4.5%) | 0.955 | 0.979 | 0.966 |
| Techno (hand-picked) | 1479 | 29 (2.0%) | 67 (4.4%) | 0.956 | 0.980 | 0.968 |
| Tekno / acid (hand-picked) | 603 | 22 (3.6%) | 13 (2.2%) | 0.978 | 0.964 | 0.971 |
| Trance / psy (hand-picked) | 648 | 6 (0.9%) | 45 (6.6%) | 0.934 | 0.991 | 0.962 |
| **All excerpts with kicks** | 42728 | 1145 (2.7%) | 1742 (4.0%) | 0.960 | 0.973 | 0.966 |

Report latency after the kick onset: median 24 ms (10th–90th percentile 14–27 ms).  
Kick-free excerpts (ambient, beatless intros, breakdowns; 35 excerpts, 22 min): 29 false triggers = 1.3 per minute.  
Per excerpt (38 s scored each): 216 of 578 excerpts have zero errors, 366 have at most two; the worst 10 excerpts account for 586 of 2916 errors.

### Mode `accurate` — tempo assist off

| set | kicks | missed (FN) | false (FP) | precision | recall | F1 |
|---|---|---|---|---|---|---|
| Hard techno / schranz / hard dance (hand-picked) | 2019 | 35 (1.7%) | 43 (2.1%) | 0.979 | 0.983 | 0.981 |
| Library random sample, 128–139 BPM | 16931 | 582 (3.4%) | 671 (3.9%) | 0.961 | 0.966 | 0.963 |
| Library random sample, 140–149 BPM | 9583 | 182 (1.9%) | 334 (3.4%) | 0.966 | 0.981 | 0.973 |
| Library random sample, 150–164 BPM | 3651 | 172 (4.7%) | 302 (8.0%) | 0.920 | 0.953 | 0.936 |
| Library random sample, < 128 BPM | 6821 | 525 (7.7%) | 415 (6.2%) | 0.938 | 0.923 | 0.931 |
| Library random sample, ≥ 165 BPM | 713 | 92 (12.9%) | 71 (10.3%) | 0.897 | 0.871 | 0.884 |
| Synthwave / darkwave (hand-picked) | 280 | 8 (2.9%) | 14 (4.9%) | 0.951 | 0.971 | 0.961 |
| Techno (hand-picked) | 1479 | 34 (2.3%) | 63 (4.2%) | 0.958 | 0.977 | 0.968 |
| Tekno / acid (hand-picked) | 603 | 15 (2.5%) | 20 (3.3%) | 0.967 | 0.975 | 0.971 |
| Trance / psy (hand-picked) | 648 | 5 (0.8%) | 16 (2.4%) | 0.976 | 0.992 | 0.984 |
| **All excerpts with kicks** | 42728 | 1650 (3.9%) | 1949 (4.5%) | 0.955 | 0.961 | 0.958 |

Report latency after the kick onset: median 25 ms (10th–90th percentile 16–28 ms).  
Kick-free excerpts (ambient, beatless intros, breakdowns; 35 excerpts, 22 min): 27 false triggers = 1.2 per minute.  
Per excerpt (38 s scored each): 200 of 578 excerpts have zero errors, 336 have at most two; the worst 10 excerpts account for 615 of 3626 errors.

### Mode `fast` — tempo assist on

| set | kicks | missed (FN) | false (FP) | precision | recall | F1 |
|---|---|---|---|---|---|---|
| Hard techno / schranz / hard dance (hand-picked) | 2019 | 124 (6.1%) | 132 (6.5%) | 0.935 | 0.939 | 0.937 |
| Library random sample, 128–139 BPM | 16931 | 581 (3.4%) | 736 (4.3%) | 0.957 | 0.966 | 0.961 |
| Library random sample, 140–149 BPM | 9583 | 331 (3.5%) | 293 (3.1%) | 0.969 | 0.965 | 0.967 |
| Library random sample, 150–164 BPM | 3651 | 203 (5.6%) | 249 (6.7%) | 0.933 | 0.944 | 0.938 |
| Library random sample, < 128 BPM | 6821 | 587 (8.6%) | 327 (5.0%) | 0.950 | 0.914 | 0.932 |
| Library random sample, ≥ 165 BPM | 713 | 92 (12.9%) | 57 (8.4%) | 0.916 | 0.871 | 0.893 |
| Synthwave / darkwave (hand-picked) | 280 | 16 (5.7%) | 11 (4.0%) | 0.960 | 0.943 | 0.951 |
| Techno (hand-picked) | 1479 | 67 (4.5%) | 157 (10.0%) | 0.900 | 0.955 | 0.927 |
| Tekno / acid (hand-picked) | 603 | 37 (6.1%) | 13 (2.2%) | 0.978 | 0.939 | 0.958 |
| Trance / psy (hand-picked) | 648 | 12 (1.9%) | 23 (3.5%) | 0.965 | 0.981 | 0.973 |
| **All excerpts with kicks** | 42728 | 2050 (4.8%) | 1998 (4.7%) | 0.953 | 0.952 | 0.953 |

Report latency after the kick onset: median 9 ms (10th–90th percentile 3–12 ms).  
Kick-free excerpts (ambient, beatless intros, breakdowns; 35 excerpts, 22 min): 50 false triggers = 2.3 per minute.  
Per excerpt (38 s scored each): 181 of 578 excerpts have zero errors, 321 have at most two; the worst 10 excerpts account for 773 of 4098 errors.

### Mode `fast` — tempo assist off

| set | kicks | missed (FN) | false (FP) | precision | recall | F1 |
|---|---|---|---|---|---|---|
| Hard techno / schranz / hard dance (hand-picked) | 2019 | 121 (6.0%) | 145 (7.1%) | 0.929 | 0.940 | 0.935 |
| Library random sample, 128–139 BPM | 16931 | 649 (3.8%) | 1202 (6.9%) | 0.931 | 0.962 | 0.946 |
| Library random sample, 140–149 BPM | 9583 | 362 (3.8%) | 442 (4.6%) | 0.954 | 0.962 | 0.958 |
| Library random sample, 150–164 BPM | 3651 | 227 (6.2%) | 364 (9.6%) | 0.904 | 0.938 | 0.921 |
| Library random sample, < 128 BPM | 6821 | 675 (9.9%) | 665 (9.8%) | 0.902 | 0.901 | 0.902 |
| Library random sample, ≥ 165 BPM | 713 | 82 (11.5%) | 82 (11.5%) | 0.885 | 0.885 | 0.885 |
| Synthwave / darkwave (hand-picked) | 280 | 16 (5.7%) | 17 (6.0%) | 0.940 | 0.943 | 0.941 |
| Techno (hand-picked) | 1479 | 76 (5.1%) | 190 (11.9%) | 0.881 | 0.949 | 0.913 |
| Tekno / acid (hand-picked) | 603 | 34 (5.6%) | 19 (3.2%) | 0.968 | 0.944 | 0.955 |
| Trance / psy (hand-picked) | 648 | 10 (1.5%) | 24 (3.6%) | 0.964 | 0.985 | 0.974 |
| **All excerpts with kicks** | 42728 | 2252 (5.3%) | 3150 (7.2%) | 0.928 | 0.947 | 0.937 |

Report latency after the kick onset: median 9 ms (10th–90th percentile 4–13 ms).  
Kick-free excerpts (ambient, beatless intros, breakdowns; 35 excerpts, 22 min): 44 false triggers = 2.0 per minute.  
Per excerpt (38 s scored each): 157 of 578 excerpts have zero errors, 274 have at most two; the worst 10 excerpts account for 874 of 5446 errors.

## Fresh hold-out (shipped weights, shipped code)

24 further tracks that were never part of training or cross-validation were run through the exported
weights with `kickdetect/kick_detector.py` (`tools/holdout.py`). 23 excerpts from 15 tracks got a trusted
annotation; in one of them the annotation sits ~120 ms ahead of the visible kick sweeps (annotation fault,
verified by eye), so both views are shown for the default configuration.

| configuration | kicks | missed | false | precision | recall | F1 |
|---|---|---|---|---|---|---|
| `accurate`, tempo assist on — 22 excerpts | 1713 | 78 (4.6%) | 56 (3.3%) | 0.967 | 0.954 | 0.961 |
| `accurate`, tempo assist on — all 23 incl. the faulty annotation | 1819 | 180 (9.9%) | 148 (8.3%) | 0.917 | 0.901 | 0.909 |
| `accurate`, tempo assist off — 22 excerpts | 1713 | 238 (13.9%) | 71 (4.6%) | 0.954 | 0.861 | 0.905 |
| `fast`, tempo assist on — 22 excerpts | 1713 | 124 (7.2%) | 54 (3.3%) | 0.967 | 0.928 | 0.947 |
| `fast`, tempo assist off — 22 excerpts | 1713 | 212 (12.4%) | 101 (6.3%) | 0.937 | 0.876 | 0.906 |

13 of the 22 excerpts have at most 3 errors in the default configuration and 20 have fewer than 10; one experimental 121 BPM track accounts for 57 of the 134 errors. The gap between tempo assist
on and off comes almost entirely from one hard-techno track whose sub band is saturated continuously
(147 of the 238 misses): the network alone stays just under threshold there, and the tempo prior lifts
the on-beat kicks over it. That is the main reason tempo assist is on by default.

## Robustness (shipped `accurate` weights, tempo assist on, 16 excerpts)

| input condition | precision | recall | F1 |
|---|---|---|---|
| reference (44.1 kHz, as mastered) | 0.989 | 0.956 | 0.972 |
| gain -30 dB | 0.989 | 0.956 | 0.972 |
| gain +6 dB with hard clipping | 0.989 | 0.957 | 0.972 |
| resampled to 48 kHz | 0.989 | 0.955 | 0.972 |
| pitched +6% (DJ pitch fader) | 0.988 | 0.953 | 0.970 |
| pitched -6% (DJ pitch fader) | 0.990 | 0.950 | 0.970 |
| low-pass 12 kHz (lossy source) | 0.989 | 0.957 | 0.972 |
| high-pass 60 Hz (small PA / club EQ cut) | 0.992 | 0.946 | 0.969 |
| added broadband noise at -30 dB | 0.989 | 0.957 | 0.972 |

These excerpts are part of the training set, so read the rows relative to the first one: the detector
is insensitive to level, clipping, sample rate, ±6 % pitch-fader changes, lossy high-frequency loss,
a 60 Hz high-pass and added noise.

## Implementation agreement

- Python detector vs. the PyTorch training pipeline: max activation difference 1.3e-6.
- C vs. Python on the test vector: max activation difference 1.3e-3 (float32 vs. float64 filters, at
  the steep edges of the activation), identical events (`tests/check_c.py`).
- C speed: about 31× real time on one core of an Apple M1 (-O2, no SIMD), i.e. ~3 % of a core.

## Known limitations

- **Annotation ceiling.** A substantial share of the remaining "errors" in the ~25 worst excerpts I inspected were
  annotation issues (off-grid kicks the reference rejects, grace notes, sparse kicks in "kick-free"
  excerpts). The true error rates are therefore somewhat lower than reported, but I cannot say by how
  much without hand-labelled data.
- **Masked kicks.** A loud tonal pad or mid-bass synth sitting on 150–500 Hz can hide the punch of the
  kick; the network then misses kicks that the tempo assist only partly recovers.
- **Long 808-style sub tails with tremolo** can re-trigger the detector inside the tail (seen in
  electro). Raising the threshold or lengthening the 60 ms refractory period helps.
- **Sub-less bass stabs** (short 100–250 Hz hits with no sub) occasionally trigger it before the real
  kick enters.
- **Very fast material (≥ 165 BPM)** is thin in this library (713 kicks in the benchmark) and scores
  lowest (F1 0.91). Hardcore / gabber / frenchcore would need more data of that kind.
- **Non-4/4 styles** (electro, breaks, IDM) look right on inspection in most excerpts but are not
  covered by trusted numbers. Tempo assist assumes a repeating pattern of 105–215 BPM; on free-form
  material consider switching it off.
- **Tempo assist needs ~2–4 beats** to lock; the first kicks after a long break are judged by the
  network alone (measured: recall on kicks that follow ≥ 1.5 s without a kick is about 86 % with or
  without tempo assist, vs. 96–97 % overall).
- **Not tested:** two tracks overlapping in a DJ transition, live microphone input, vinyl with heavy
  rumble, sample rates below 44.1 kHz.

## Reproducing

```
python tools/build_corpus.py                      # hand-picked style groups + first random sample
python tools/extend_corpus.py 170 99              # more random tracks (count, seed)
python tools/extend_corpus.py 250 4242
python tools/separate.py                          # Demucs stems (GPU/MPS, ~6 s per excerpt)
python tools/reference.py                         # offline annotations + trust flags -> data/ref, data/ref_meta.json
python tools/nn.py cnn 30:0 --epochs 8 --tag _plain   # 3-fold cross-validation by track, accurate mode
python tools/nn.py cnn 14:0 --epochs 8 --tag _plain   # ... fast mode
python tools/tempo_eval.py data/acts_cnn_30_plain.npz 30   # tempo-assist strength / threshold sweep
python tools/report.py accurate=data/acts_cnn_30_plain.npz:30:0.97:0.985 fast=data/acts_cnn_14_plain.npz:14:0.97:0.993
python tools/final.py --lat 30 --thr 0.97 --thr-tempo 0.985 --name accurate --epochs 8
python tools/final.py --lat 14 --thr 0.97 --thr-tempo 0.993 --name fast --epochs 8
python tests/make_vectors.py && python tests/check_c.py
python tools/extend_corpus.py 24 777 holdout && python tools/separate.py && python tools/holdout.py
python tools/robust.py accurate 16
```

`tools/plotf.py out.png <excerpt> <t0> <t1> …` draws the network's band image with reference and
detections, which is how the worst excerpts were inspected.
