// Microphone capture: resamples the context rate (usually 48 kHz) to 16 kHz mono and
// posts Int16 frames of 512 samples (32 ms) - the frame size the server's Silero VAD uses.
// Also posts an RMS level for the meter.
//
// Resampling = anti-alias low-pass FIR (windowed sinc, 7 kHz cutoff) + linear interpolation.
// A plain average of neighbouring samples is not enough: sibilants (s, sh, f, th) carry
// energy above 8 kHz that would fold back into the speech band and confuse Whisper.

const OUT_RATE = 16000;
const TAPS = 63;
const CUTOFF_HZ = 7000;

function designLowpass(inRate) {
  const h = new Float32Array(TAPS);
  const fc = CUTOFF_HZ / inRate; // cycles per sample
  const m = (TAPS - 1) / 2;
  let sum = 0;
  for (let i = 0; i < TAPS; i++) {
    const n = i - m;
    const sinc = n === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * n) / (Math.PI * n);
    // Blackman window: ~ -74 dB sidelobes
    const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (TAPS - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (TAPS - 1));
    h[i] = sinc * w;
    sum += h[i];
  }
  for (let i = 0; i < TAPS; i++) h[i] /= sum; // unity gain at DC
  return h;
}

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / OUT_RATE; // input samples per output sample
    this.passthrough = sampleRate <= OUT_RATE;
    this.h = designLowpass(sampleRate);
    this.ring = new Float32Array(TAPS);
    this.ri = 0;
    this.yPrev = 0;
    this.inIndex = 0; // index of the current input sample
    this.nextOut = 0; // position of the next output sample, in input-sample units
    this.buf = new Int16Array(512);
    this.n = 0;
    this.levelAcc = 0;
    this.levelN = 0;
  }

  emit(v) {
    this.buf[this.n++] = Math.max(-1, Math.min(1, v)) * 32767;
    if (this.n === 512) {
      this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
      this.buf = new Int16Array(512);
      this.n = 0;
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const h = this.h, ring = this.ring;
    for (let i = 0; i < ch.length; i++) {
      const s = ch[i];
      this.levelAcc += s * s;
      this.levelN++;
      if (this.passthrough) {
        this.emit(s);
        continue;
      }
      // FIR low-pass over the last TAPS input samples (circular buffer).
      ring[this.ri] = s;
      let y = 0;
      let k = this.ri;
      for (let t = 0; t < TAPS; t++) {
        y += h[t] * ring[k];
        k = k === 0 ? TAPS - 1 : k - 1;
      }
      this.ri = this.ri === TAPS - 1 ? 0 : this.ri + 1;
      // Linear interpolation between filtered samples (inIndex-1, inIndex).
      while (this.nextOut <= this.inIndex) {
        const frac = this.nextOut - (this.inIndex - 1);
        this.emit(this.yPrev + (y - this.yPrev) * frac);
        this.nextOut += this.ratio;
      }
      this.yPrev = y;
      this.inIndex++;
    }
    if (this.levelN >= 4096) {
      this.port.postMessage({ level: Math.sqrt(this.levelAcc / this.levelN) });
      this.levelAcc = 0;
      this.levelN = 0;
    }
    return true;
  }
}
registerProcessor("capture-processor", CaptureProcessor);
