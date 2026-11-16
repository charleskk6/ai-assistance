// Tests the microphone resampler in public/capture-worklet.js outside a browser.
//   node --test frontend/tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const src = fs.readFileSync(path.join(here, "..", "public", "capture-worklet.js"), "utf8");

function makeProcessor(rate) {
  let Proc;
  const posted = [];
  const ctx = {
    sampleRate: rate,
    Math, Float32Array, Int16Array,
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage: (m) => posted.push(m) }; } },
    registerProcessor: (_name, cls) => { Proc = cls; },
  };
  vm.createContext(ctx);
  vm.runInContext(src, ctx);
  return { proc: new Proc(), posted };
}

function run(rate, freq, seconds = 1, amp = 0.5) {
  const { proc, posted } = makeProcessor(rate);
  const total = Math.round(rate * seconds);
  for (let off = 0; off < total; off += 128) {
    const block = new Float32Array(128);
    for (let i = 0; i < 128; i++) block[i] = amp * Math.sin((2 * Math.PI * freq * (off + i)) / rate);
    proc.process([[block]]);
  }
  const frames = posted.filter((m) => m instanceof ArrayBuffer).map((b) => new Int16Array(b));
  const out = Float32Array.from(frames.flatMap((f) => Array.from(f)).map((v) => v / 32767));
  return out;
}

const rms = (a, skip = 400) => Math.sqrt(a.slice(skip).reduce((s, v) => s + v * v, 0) / (a.length - skip));
const db = (x) => 20 * Math.log10(x);

for (const rate of [48000, 44100]) {
  test(`${rate} Hz: output is 16 kHz and in 512-sample frames`, () => {
    const out = run(rate, 440, 2);
    // 2 s in -> ~32000 samples out (frames are only posted when full)
    assert.ok(Math.abs(out.length - 32000) <= 512, `got ${out.length}`);
    assert.equal(out.length % 512, 0);
  });

  test(`${rate} Hz: speech band passes almost unchanged`, () => {
    for (const f of [300, 1000, 3000, 5000]) {
      const gain = db(rms(run(rate, f)) / (0.5 / Math.SQRT2));
      assert.ok(gain > -1.0 && gain < 0.5, `${f} Hz gain ${gain.toFixed(2)} dB`);
    }
  });

  test(`${rate} Hz: content above 8 kHz does not alias into the speech band`, () => {
    for (const f of [9000, 10000, 12000, 15000]) {
      const gain = db(rms(run(rate, f)) / (0.5 / Math.SQRT2));
      assert.ok(gain < -40, `${f} Hz leaks at ${gain.toFixed(1)} dB`);
    }
  });
}
