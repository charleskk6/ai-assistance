// Microphone capture: decimates the context rate (usually 48 kHz) to 16 kHz mono
// with a box filter, and posts Int16 frames of 512 samples (32 ms) - the frame
// size the server's Silero VAD uses. Also posts an RMS level for the meter.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;
    this.acc = 0;
    this.accN = 0;
    this.buf = new Int16Array(512);
    this.n = 0;
    this.levelAcc = 0;
    this.levelN = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const s = ch[i];
      this.levelAcc += s * s;
      this.levelN++;
      this.acc += s;
      this.accN++;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.accN));
        this.acc = 0;
        this.accN = 0;
        this.buf[this.n++] = v * 32767;
        if (this.n === 512) {
          this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
          this.buf = new Int16Array(512);
          this.n = 0;
        }
      }
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
