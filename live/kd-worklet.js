// AudioWorklet: runs the kick detector on the audio thread and posts display data to the page.
import { KickDetector } from "../kickdetect/kick_detector.js";

const BATCH = 6;        // render quanta per message (~16 ms)

class KickWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions;
    this.weights = o.weights;
    this.cfg = { mode: o.mode, tempo: o.tempo, threshold: null };
    this.mono = new Float32Array(128);
    this.block = 0;
    this.n = 0;
    this.alloc();
    this.build();
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.mode !== undefined && m.mode !== this.cfg.mode) { this.cfg.mode = m.mode; this.cfg.threshold = null; this.build(); }
      if (m.tempo !== undefined) { this.cfg.tempo = m.tempo; this.cfg.threshold = null; this.det.setTempoAssist(m.tempo); }
      if (m.threshold !== undefined) { this.cfg.threshold = m.threshold; this.det.threshold = m.threshold; }
      if (m.reset) this.build();
      this.info();
    };
  }

  alloc() {
    this.mn = new Float32Array(BATCH); this.mx = new Float32Array(BATCH);
    this.act = new Float32Array(BATCH); this.raw = new Float32Array(BATCH);
    this.events = [];
  }

  build() {
    this.det = new KickDetector(sampleRate, this.weights[this.cfg.mode], { tempoAssist: this.cfg.tempo });
    if (this.cfg.threshold !== null) this.det.threshold = this.cfg.threshold;
    this.offset = this.block * 128;                 // detector time 0 in stream samples
    this.fa = 0; this.fr = 0;
    this.det.onFrame = (a, r) => { if (a > this.fa) this.fa = a; if (r > this.fr) this.fr = r; };
    this.info();
  }

  info() {
    const d = this.det;
    this.port.postMessage({ info: { threshold: d.threshold, latency: d.latency, hop: d.hop, sampleRate } });
  }

  process(inputs) {
    const ch = inputs[0], mono = this.mono;
    if (ch && ch.length) {
      const g = 1 / ch.length;
      for (let i = 0; i < 128; i++) { let s = 0; for (let c = 0; c < ch.length; c++) s += ch[c][i]; mono[i] = s * g; }
    } else mono.fill(0);
    let lo = 0, hi = 0;
    for (let i = 0; i < 128; i++) { const s = mono[i]; if (s < lo) lo = s; else if (s > hi) hi = s; }
    const ev = this.det.process(mono);
    for (const e of ev) this.events.push({ onset: e.onsetTime * sampleRate + this.offset, at: e.time * sampleRate + this.offset, strength: e.strength });
    const k = this.n++;
    this.mn[k] = lo; this.mx[k] = hi;
    // peak over the frames that ended in this quantum; hold the last value if none did (hop > 128)
    if (this.fa < 0) { this.fa = this.det.activation; this.fr = this.det.rawActivation; }
    this.act[k] = this.fa; this.raw[k] = this.fr;
    this.fa = -1; this.fr = -1;
    this.block++;
    if (this.n === BATCH) {
      this.port.postMessage({ b0: this.block - BATCH, mn: this.mn, mx: this.mx, act: this.act, raw: this.raw, events: this.events },
        [this.mn.buffer, this.mx.buffer, this.act.buffer, this.raw.buffer]);
      this.alloc(); this.n = 0;
    }
    return true;
  }
}
registerProcessor("kick-worklet", KickWorklet);
