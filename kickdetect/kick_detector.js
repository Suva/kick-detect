// Real-time kick drum detector for dance music — JavaScript implementation (ES module, no dependencies).
// Follows docs/ALGORITHM.md step by step; same weights JSON as the Python version.
//
//   const det = new KickDetector(sampleRate, weightsJson, { tempoAssist: true });
//   const events = det.process(float32MonoBlock);   // [{time, onsetTime, strength}], times in seconds
//
// No allocation happens in process() except for the event objects, so it can run in an AudioWorklet.

const FC = [45, 64, 90, 128, 180, 256, 360, 512, 720, 1024, 2048, 4096, 8192];
const QS = [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1];
const NB = 13, KT = 5, HOP_HZ = 500, TAU = 4.0, EPS = 1e-3, REFRACTORY = 0.060;
const NONE = [];

export class KickDetector {
  constructor(sampleRate, weights, opts = {}) {
    const L = {};
    for (const [k, v] of Object.entries(weights.layers)) L[k] = Float32Array.from(v.data);
    const sh = (k) => weights.layers[k].shape;
    this.w1 = L["c1.weight"]; this.b1 = L["c1.bias"]; this.c1 = sh("c1.weight")[0];
    this.w2 = L["c2.weight"]; this.b2 = L["c2.bias"]; this.c2 = sh("c2.weight")[0];
    this.w3 = L["c3.weight"]; this.b3 = L["c3.bias"]; this.c3 = sh("c3.weight")[0];
    this.wd1 = L["d1.weight"]; this.bd1 = L["d1.bias"]; this.hid = sh("d1.weight")[0];
    this.wd2 = L["d2.weight"]; this.bd2 = L["d2.bias"][0];
    this.latFrames = weights.lat_frames | 0;
    this.beta = weights.tempo_beta ?? 3.0;
    this.thresholdPlain = weights.threshold;
    this.thresholdTempo = weights.threshold_tempo ?? weights.threshold;
    this.tempoAssist = opts.tempoAssist !== false;
    this.threshold = opts.threshold ?? (this.tempoAssist ? this.thresholdTempo : this.thresholdPlain);

    const sr = this.sr = sampleRate;
    const hop = this.hop = Math.round(sr / HOP_HZ);
    const fr = this.frameRate = sr / hop;
    this.latency = this.latFrames / fr;

    // filterbank: two cascaded band-pass biquads per band (transposed direct form II)
    this.cb0 = new Float64Array(NB); this.ca1 = new Float64Array(NB); this.ca2 = new Float64Array(NB);
    for (let b = 0; b < NB; b++) {
      const w0 = 2 * Math.PI * FC[b] / sr, al = Math.sin(w0) / (2 * QS[b]), a0 = 1 + al;
      this.cb0[b] = al / a0; this.ca1[b] = -2 * Math.cos(w0) / a0; this.ca2[b] = (1 - al) / a0;
    }
    this.z = new Float64Array(NB * 4);
    this.pk = new Float64Array(NB);
    this.hopCount = 0;
    this.hold = []; this.hpos = new Int32Array(NB);
    for (let b = 0; b < NB; b++) this.hold.push(new Float64Array(Math.ceil(0.5 / (FC[b] * hop / sr))));
    this.dec = Math.exp(-1 / (TAU * fr));
    this.g = 1e-4;
    this.r = new Float64Array(NB).fill(1e-4);
    this.C = new Float32Array(NB);

    // network histories, oldest first
    const mk = (n, len) => Array.from({ length: n }, () => new Float32Array(len));
    this.x0 = mk(KT, 2 * NB);
    this.h1 = mk(2 * (KT - 1) + 1, this.c1 * (NB - 2));
    this.h2 = mk(4 * (KT - 1) + 1, this.c2 * (NB - 4));
    this.h3 = new Float32Array(this.c3 * (NB - 6));
    this.taps = new Array(KT);
    this.feat = new Float32Array(this.c3 * (NB - 6) + NB);

    // tempo assist: comb bank, one comb per beat period (105..215 BPM)
    const p0 = Math.floor(fr * 60 / 215), p1 = Math.floor(fr * 60 / 105);
    this.nP = Math.min(p1 - p0 + 1, 160);
    this.cP = new Int32Array(this.nP); this.calpha = new Float32Array(this.nP);
    for (let i = 0; i < this.nP; i++) { this.cP[i] = p0 + i; this.calpha[i] = Math.pow(0.5, (p0 + i) / (2.5 * fr)); }
    this.cL = this.cP[this.nP - 1] + 1;
    this.cbuf = new Float32Array(this.nP * this.cL);
    this.cscore = new Float32Array(this.nP);
    this.ci = 0; this.cpk = 1e-3;
    this.csdec = Math.exp(-1 / (3 * fr)); this.cpdec = Math.exp(-1 / (4 * fr));

    this.refractory = Math.floor(REFRACTORY * fr + 0.5);
    this.sinceLast = 1e9;
    this.nsamples = 0;
    this.activation = 0; this.rawActivation = 0;
    this.onFrame = null;            // optional: called as onFrame(activation, rawActivation) every hop
  }

  setTempoAssist(on) {
    this.tempoAssist = !!on;
    this.threshold = on ? this.thresholdTempo : this.thresholdPlain;
  }

  // valid conv over bands + ReLU for the newest time column; taps = KT frames oldest..newest, each [cin][win]
  _conv(taps, w, bias, cin, cout, win, out) {
    const wd = win - 2;
    for (let o = 0; o < cout; o++) {
      for (let x = 0; x < wd; x++) {
        let s = bias[o];
        for (let c = 0; c < cin; c++) {
          const off = c * win + x;
          let wi = (o * cin + c) * KT * 3;
          for (let t = 0; t < KT; t++, wi += 3) {
            const f = taps[t];
            s += w[wi] * f[off] + w[wi + 1] * f[off + 1] + w[wi + 2] * f[off + 2];
          }
        }
        out[o * wd + x] = s > 0 ? s : 0;
      }
    }
  }

  _frame() {
    const { pk, hold, hpos, r, C, taps } = this;
    // envelope with half-period hold, slow peak followers
    let emax = 0;
    for (let b = 0; b < NB; b++) {
      const h = hold[b]; h[hpos[b]] = pk[b]; hpos[b] = (hpos[b] + 1) % h.length; pk[b] = 0;
      let e = 0; for (let i = 0; i < h.length; i++) if (h[i] > e) e = h[i];
      pk[b] = e; if (e > emax) emax = e;            // pk temporarily holds E
    }
    const g = this.g = Math.max(this.g * this.dec, emax);
    const x = this.x0.shift();
    for (let b = 0; b < NB; b++) {
      const e = pk[b]; pk[b] = 0;
      r[b] = Math.max(r[b] * this.dec, e);
      const F = 1 + Math.log10((e + EPS * g) / g) / 1.5;
      C[b] = 1 + Math.log10((r[b] + EPS * g) / g) / 1.5;
      x[b] = F; x[NB + b] = F - C[b];
    }
    this.x0.push(x);
    // network
    const o1 = this.h1.shift();
    this._conv(this.x0, this.w1, this.b1, 2, this.c1, NB, o1);
    this.h1.push(o1);
    for (let t = 0; t < KT; t++) taps[t] = this.h1[2 * t];
    const o2 = this.h2.shift();
    this._conv(taps, this.w2, this.b2, this.c1, this.c2, NB - 2, o2);
    this.h2.push(o2);
    for (let t = 0; t < KT; t++) taps[t] = this.h2[4 * t];
    this._conv(taps, this.w3, this.b3, this.c2, this.c3, NB - 4, this.h3);
    const feat = this.feat, n3 = this.h3.length, nf = feat.length;
    feat.set(this.h3); feat.set(C, n3);
    let zz = this.bd2;
    for (let j = 0; j < this.hid; j++) {
      let s = this.bd1[j]; const wo = j * nf;
      for (let i = 0; i < nf; i++) s += this.wd1[wo + i] * feat[i];
      if (s > 0) zz += this.wd2[j] * s;
    }
    const raw = 1 / (1 + Math.exp(-zz));
    this.rawActivation = raw;
    return this._tempo(raw);
  }

  _tempo(a) {
    const { nP, cP, cL, cbuf, cscore, calpha } = this;
    let best = 0, bs = -Infinity;
    for (let i = 0; i < nP; i++) if (cscore[i] > bs) { bs = cscore[i]; best = i; }
    const rd = (i) => cbuf[i * cL + (((this.ci - cP[i]) % cL) + cL) % cL];
    const pred = rd(best);
    this.kickPeriod = cP[best] / this.frameRate;
    this.cpk = Math.max(this.cpk * this.cpdec, pred);
    const prior = pred / Math.max(this.cpk, 0.05);
    const wr = this.ci % cL;
    for (let i = 0; i < nP; i++) {
      const prev = rd(i);
      cscore[i] = cscore[i] * this.csdec + a * prev;
      cbuf[i * cL + wr] = (1 - calpha[i]) * a + calpha[i] * prev;
    }
    this.ci++;
    if (!this.tempoAssist) return a;
    const ac = Math.min(Math.max(a, 1e-6), 1 - 1e-6);
    return 1 / (1 + Math.exp(-(Math.log(ac / (1 - ac)) + this.beta * (2 * prior - 1))));
  }

  /** Feed mono samples (any block size). Returns the kicks reported in this block. */
  process(x, n = x.length) {
    const { cb0, ca1, ca2, z, pk, hop } = this;
    let events = NONE;
    for (let i = 0; i < n; i++) {
      const s = x[i];
      for (let b = 0, k = 0; b < NB; b++, k += 4) {
        const b0 = cb0[b], a1 = ca1[b], a2 = ca2[b];
        let y = b0 * s + z[k];
        z[k] = z[k + 1] - a1 * y; z[k + 1] = -b0 * s - a2 * y;
        const y2 = b0 * y + z[k + 2];
        z[k + 2] = z[k + 3] - a1 * y2; z[k + 3] = -b0 * y - a2 * y2;
        y = y2 < 0 ? -y2 : y2;
        if (y > pk[b]) pk[b] = y;
      }
      this.nsamples++;
      if (++this.hopCount === hop) {
        this.hopCount = 0;
        for (let k = 0; k < z.length; k++) if (z[k] < 1e-25 && z[k] > -1e-25) z[k] = 0;   // no denormals in silence
        const a = this.activation = this._frame();
        if (this.onFrame) this.onFrame(a, this.rawActivation);
        this.sinceLast++;
        if (a >= this.threshold && this.sinceLast >= this.refractory) {
          this.sinceLast = 0;
          const t = this.nsamples / this.sr;
          if (events === NONE) events = [];
          events.push({ time: t, onsetTime: t - this.latency, strength: a });
        }
      }
    }
    return events;
  }
}
