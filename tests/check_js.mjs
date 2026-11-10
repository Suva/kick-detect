// Compare the JavaScript implementation with the stored vectors:  node tests/check_js.mjs
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { KickDetector } from "../kickdetect/kick_detector.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const V = join(root, "tests/vectors");
const meta = JSON.parse(readFileSync(join(V, "clip.json"), "utf8"));
const raw = readFileSync(join(V, "clip.f32"));
const x = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
const nums = (f) => readFileSync(join(V, f), "utf8").trim().split("\n").map((l) => parseFloat(l));
let ok = true;
for (const mode of ["accurate", "fast"]) {
  const w = JSON.parse(readFileSync(join(root, `kickdetect/weights_${mode}.json`), "utf8"));
  for (const ta of [true, false]) {
    const tag = `${mode}_${ta ? "tempo" : "notempo"}`;
    const det = new KickDetector(meta.sample_rate, w, { tempoAssist: ta });
    const acts = []; det.onFrame = (a) => acts.push(a);
    const ev = [];
    for (let i = 0; i < x.length; i += 128) ev.push(...det.process(x.subarray(i, Math.min(i + 128, x.length))));
    const ref = nums(`activations_${tag}.txt`), te = nums(`events_${tag}.txt`);
    let d = 0;
    for (let i = 100; i < Math.min(acts.length, ref.length); i++) d = Math.max(d, Math.abs(acts[i] - ref[i]));
    const same = ev.length === te.length && ev.every((e, i) => Math.abs(e.time - te[i]) < 0.0031);
    const good = d < 2e-3 && same && acts.length === ref.length; ok &&= good;
    console.log(`${tag.padEnd(18)} frames=${acts.length} max|JS-ref|=${d.toExponential(2)} events JS=${ev.length} ref=${te.length} ${good ? "OK" : "MISMATCH"}`);
  }
}
process.exit(ok ? 0 : 1);
