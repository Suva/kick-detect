# Kick detector — algorithm specification

This is the language-neutral description of the detector. Everything here is plain arithmetic on
floats: 26 biquad filters, a few max/log operations and a ~10k-weight convolutional network. The
C (`kickdetect/kick_detector.h`) and Python (`kickdetect/kick_detector.py`) implementations follow
this document step by step and produce the same events and activations within ~1e-3.

Input: mono audio (sum L+R and halve), float, nominal range [-1, 1], any sample rate ≥ 32 kHz
(tested at 44.1 kHz and 48 kHz). All state is carried between blocks, so the host block size does
not matter.

## 1. Filterbank

13 band-pass bands. Each band is **two identical cascaded biquads** (RBJ "constant 0 dB peak gain"
band-pass):

| band | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| centre fc (Hz) | 45 | 64 | 90 | 128 | 180 | 256 | 360 | 512 | 720 | 1024 | 2048 | 4096 | 8192 |
| Q | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 | 1 |

```
w0 = 2*pi*fc/sr        alpha = sin(w0)/(2*Q)       a0 = 1 + alpha
b0 = alpha/a0   b1 = 0   b2 = -alpha/a0   a1 = -2*cos(w0)/a0   a2 = (1 - alpha)/a0
y[n] = b0*x[n] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]          (applied twice per band)
```

Half-octave spacing below 1 kHz is what makes a kick's downward pitch sweep visible as a diagonal
ridge across bands; octave spacing above is enough to see the click.

## 2. Envelope per hop

`hop = round(sr / 500)` samples (88 at 44.1 kHz, 96 at 48 kHz → one frame every ~2 ms).

For every band, per hop:

1. `p` = max of `|y|` over the hop's samples.
2. Push `p` into a ring of length `H = ceil(0.5 / (fc * hop / sr))` frames
   (44.1 kHz: 6, 4, 3, 2, 2, 1, 1, …) and take `E = max(ring)`.

The hold spans half a period of the band's centre frequency, so a steady tone gives a flat
envelope, while an attack shows up in the very hop it happens (no smoothing delay).

## 3. Level normalisation

Two kinds of slow peak followers, decay `d = exp(-1 / (4 s * frame_rate))` per frame:

```
g    = max(g * d, max_b E[b])          (global; initial 1e-4)
r[b] = max(r[b] * d, E[b])             (per band; initial 1e-4)
F[b] = 1 + log10((E[b] + 0.001*g) / g) / 1.5       # band level vs. recent global peak, about [-1, 1]
C[b] = 1 + log10((r[b] + 0.001*g) / g) / 1.5       # band's own recent peak vs. global peak
```

`F` makes the detector independent of playback level; `F - C` tells the network how loud a band is
compared with the loudest thing that band did in the last few seconds (kicks sit at ≈ 0 there).

## 4. Network

Input per frame: 2 channels × 13 bands — `F[b]` and `F[b] - C[b]`.
Three causal convolutions over (time × band), kernel 5 × 3, no padding, ReLU after each:

| layer | in → out channels | time taps (frames back from now) | band width out |
|---|---|---|---|
| conv1 | 2 → C1 | 4, 3, 2, 1, 0 | 11 |
| conv2 | C1 → C2 | 8, 6, 4, 2, 0 | 9 |
| conv3 | C2 → C3 | 16, 12, 8, 4, 0 | 7 |

Kernel time index 0 is the **oldest** tap. Because each layer only needs its own newest column,
a streaming implementation keeps short histories (5 input frames, 9 conv1 frames, 17 conv2 frames)
and computes one column per layer per hop. The receptive field is 29 frames (58 ms).

Head: `v = concat(conv3 column flattened as [channel][band], C[0..12])`,
`activation = sigmoid(w2 · relu(W1 · v + b1) + b2)`.

The network is trained so that the activation peaks `lat_frames` frames after a kick's onset
(`lat_frames` is stored with the weights). That delay is the detector's latency: it is the amount
of the kick the network is allowed to see before committing.

## 5. Tempo assist (optional, on by default)

A bank of comb filters on the raw activation `a`, one per beat period `P` (in frames) from
`floor(frame_rate*60/215)` to `floor(frame_rate*60/105)` (139 … 286 at 44.1 kHz), each with a ring
buffer `buf[P]` and a score:

```
alpha[P] = 0.5 ^ (P / (2.5 s * frame_rate))          # ~2.5 s memory half-life for every period
per frame, before writing the new values:
    prev[P] = buf[P] one period ago                   # i.e. buf[P][n - P]
    best    = argmax_P score[P]                       # current beat period (lowest index wins ties)
    pred    = prev[best]
    pk      = max(pk * exp(-1/(4 s * frame_rate)), pred)        (initial 1e-3)
    prior   = pred / max(pk, 0.05)                    # 0..1: "was there a kick here on previous beats?"
    score[P] = score[P] * exp(-1/(3 s * frame_rate)) + a * prev[P]
    buf[P][n] = (1 - alpha[P]) * a + alpha[P] * prev[P]
activation = sigmoid(logit(clip(a, 1e-6, 1 - 1e-6)) + 3.0 * (2 * prior - 1))
```

The bank is fed with the *raw* network activation only (no feedback from the final decision), so it
cannot talk itself into a pattern. With tempo assist off, `activation = a`.

## 6. Event logic

A kick is reported on the first frame where `activation >= threshold` and at least 60 ms have
passed since the previous report. Two thresholds are stored with the weights: `threshold` (network
only) and `threshold_tempo` (with tempo assist). Reported time = end of the current hop; onset
estimate = reported time − `lat_frames * hop / sr`.

## 7. Verifying a port

`tests/vectors/` holds a 12 s clip (`clip.f32`, raw mono float32, 44.1 kHz) with the expected
per-frame activations (`activations_<mode>_<tempo|notempo>.txt`) and events
(`events_<mode>_<tempo|notempo>.txt`: report time, onset time, strength). A port is correct when,
after the first 0.2 s, its activations stay within about 2e-3 of the reference (float32 vs. float64
filter arithmetic accounts for differences of that size at the steep edges) and it reports the same
events. `tests/check_c.py` does this for the C implementation.
