# Real-time kick detector for dance music

A causal (no look-ahead beyond a fixed, small latency) kick drum detector tuned and validated on a
DJ library of techno, hard techno / schranz, tekno, trance, industrial and electro.

- **Input:** mono audio stream, any block size, 44.1 or 48 kHz.
- **Output:** one event per kick, reported a fixed about 25 ms (`accurate` mode) or 10 ms (`fast` mode) after the kick's onset.
- **Cost:** 26 biquads + a 10k-weight convolutional network evaluated every 2 ms — about 3 % of
  one laptop core in plain C, no SIMD, no FFT, no dependencies.
- **Portable:** the whole algorithm is specified in [docs/ALGORITHM.md](docs/ALGORITHM.md) and
  implemented twice — C99 single header and Python/numpy — with test vectors to verify a port.

## Measured accuracy

Cross-validated by track on 543 excerpts with 44,846 annotated kicks (plus 35 kick-free excerpts);
every excerpt is scored by a model that never saw its track.

| mode | median latency | kicks missed | false detections | precision | recall | F1 |
|---|---|---|---|---|---|---|
| `accurate` + tempo assist (default) | 24 ms | 2.7 % | 4.0 % | 0.960 | 0.973 | 0.966 |
| `accurate`, network only | 25 ms | 3.9 % | 4.5 % | 0.955 | 0.961 | 0.958 |
| `fast` + tempo assist | 9 ms | 4.8 % | 4.7 % | 0.953 | 0.952 | 0.953 |
| `fast`, network only | 9 ms | 5.3 % | 7.2 % | 0.928 | 0.947 | 0.937 |

- Hand-picked hard techno / schranz / hard dance (default mode): **1.0 % missed, 1.9 % false** (F1 0.985).
- Kick-free audio: about 1.3 false triggers per minute.
- 216 of 578 excerpts have zero errors and 366 at most two per 38 s; errors concentrate in a few tracks.
- 22 further excerpts from 15 tracks never used in development, run through the shipped code: 4.6 % missed, 3.3 % false.
- Unaffected by level, clipping, 48 kHz, ±6 % pitch fader, lossy sources, 60 Hz high-pass, added noise.

The reference annotations are automatic and conservative about off-grid hits, so part of the "false"
column is real kicks the reference left out; the error rates are upper bounds rather than exact figures.

Full tables, method and caveats: [docs/RESULTS.md](docs/RESULTS.md).

## Using it

### C

```c
#define KICK_DETECTOR_IMPLEMENTATION
#include "kick_detector.h"          /* uses kd_weights_accurate.h by default */

static kd_t kd;                     /* ~210 kB of state; keep it off the stack */
kd_init(&kd, 44100.f);

void audio_callback(const float *mono, int n) {
    kd_event ev[8];
    int count = kd_process(&kd, mono, n, ev, 8);
    for (int i = 0; i < count; i++)
        on_kick(ev[i].onset, ev[i].strength);   /* onset = sample index of the kick itself */
}
```

- Fast mode: compile with `-DKD_WEIGHTS_HEADER='"kd_weights_fast.h"'`.
- `kd_set_tempo_assist(&kd, 0)` turns the tempo prior off (pure audio evidence).
- `-DKD_NO_TEMPO` removes the tempo prior entirely and shrinks the state to ~17 kB.
- `kd.threshold` can be raised for fewer false triggers or lowered for fewer misses.
- No allocation, no locks, no I/O in `kd_process` — safe to call on the audio thread.

Command-line check: `ffmpeg -i track.mp3 -f f32le -ac 1 -ar 44100 - | ./kd_cli 44100`
(build with `cc -O2 -I kickdetect -o kd_cli kickdetect/kd_cli.c -lm`).

### Python

```python
from kickdetect.kick_detector import KickDetector
det = KickDetector(sample_rate=44100, mode="accurate")      # or mode="fast", tempo_assist=False
for block in blocks:                                        # mono float arrays, any length
    for ev in det.process(block):
        print(ev["onset_time"], ev["strength"])
```

Needs `numpy` and `scipy`; runs about 8× faster than real time.

### Live view (browser)

`live/run.sh` serves the project on `http://localhost:8377/live/` and opens it: pick an audio input
(or the built-in test clip, or an audio file) and watch the waveform, the detector activation and the
detected kicks scroll by. It runs `kickdetect/kick_detector.js` (a JavaScript port, checked against
the test vectors by `node tests/check_js.mjs`) in an AudioWorklet. To feed it what your DJ software
plays, select a loopback device (e.g. BlackHole) or an audio-interface input.

### Other languages

Port `kick_detector.h` (≈200 lines of loops) and load `kickdetect/weights_<mode>.json`
(layer shapes + flat row-major data). Then check the port against `tests/vectors/`:
feed `clip.f32` and compare per-frame activations with `activations_<mode>_<tempo|notempo>.txt`.

## How it works (short version)

1. **Filterbank** — 13 band-pass bands from 45 Hz to 8 kHz, half-octave spacing below 1 kHz.
2. **Envelopes every 2 ms** — per-band peak with a hold of half a band period: attacks appear
   immediately, steady bass does not ripple.
3. **Level normalisation** — log levels relative to a slow global peak follower and to each band's
   own slow peak follower, so the detector does not care about playback level.
4. **Network** — three small causal dilated convolutions over (time × band) and a dense head.
   In this representation a kick is a diagonal ridge: a broadband "punch" at 150–1000 Hz that
   sweeps down into the sub band within 20–100 ms. Bass notes, rumble, toms, snares and 808 tails
   do not have that shape, even when they are just as loud in the low band — which is why simple
   low-band energy/flux detectors fail on hard techno (the sub is already at maximum *before* the
   kick and often dips when it hits).
5. **Tempo assist** (optional) — a comb-filter bank on the network output learns the running kick
   period (105–215 BPM) and nudges the decision: weak on-pattern kicks are confirmed, isolated
   off-pattern triggers are suppressed. It only looks at the past.
6. **Decision** — threshold plus a 60 ms refractory period.

The latency is a deliberate part of the design: the network is allowed to see the first
30 ms (or 14 ms) of a kick before committing. At the very first
millisecond a kick and a sharp bass note are often indistinguishable; the sweep is what tells
them apart.

## How it was built and tested

- **Corpus:** 40-second excerpts cut at 30 % and 60 % of each track from 508 tracks of the
  library (hand-picked style groups plus a large random sample), 1,016 excerpts in total.
- **Ground truth:** produced offline and independently of the detector — Demucs source separation,
  punch-band onsets in the drum stem, an offline tempo grid, and a per-track kick template (kicks in
  electronic music are near-identical repeated samples). Only excerpts whose annotation is
  self-consistent (verified grid + template correlation) are used for the headline numbers;
  see [docs/RESULTS.md](docs/RESULTS.md) for what that includes and excludes.
- **Evaluation:** cross-validation by track — every excerpt is scored by a model that never saw
  any part of that track. A detection counts if it falls within −30 … +70 ms of a reference kick.

## Repository layout

```
kickdetect/   kick_detector.h, kd_weights_*.h, kd_cli.c      C implementation + weights
              kick_detector.py, weights_*.json               Python implementation + weights
docs/         ALGORITHM.md (spec), RESULTS.md (benchmarks)
tests/        vectors/ + make_vectors.py, check_c.py         port verification
tools/        corpus building, reference annotation, training, evaluation
corpus/ stems/ data/                                         audio excerpts, separated stems, caches (large, reproducible)
```

Re-training on your own material: `tools/build_corpus.py` / `extend_corpus.py` → `separate.py` →
`reference.py` → `nn.py` (cross-validation) → `final.py` (export) → `tests/make_vectors.py`.
