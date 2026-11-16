// Browser audio: microphone capture (AudioWorklet -> 16 kHz Int16 frames) and
// low-latency playback of server TTS chunks with instant, turn-aware cancellation.

export interface CaptureHandle {
  stop(): void;
  ctx: AudioContext;
}

export async function startCapture(
  onFrame: (pcm: ArrayBuffer) => void,
  onLevel: (rms: number) => void,
): Promise<CaptureHandle> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true, // browser AEC; most effective in Safari, best paired with headphones
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  const ctx = new AudioContext({ latencyHint: "interactive" });
  await ctx.audioWorklet.addModule("/capture-worklet.js");
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "capture-processor");
  node.port.onmessage = (e: MessageEvent) => {
    if (e.data instanceof ArrayBuffer) onFrame(e.data);
    else if (e.data && typeof e.data.level === "number") onLevel(e.data.level);
  };
  src.connect(node);
  // Keep the graph pulled without making the mic audible.
  const mute = ctx.createGain();
  mute.gain.value = 0;
  node.connect(mute).connect(ctx.destination);
  return {
    ctx,
    stop() {
      stream.getTracks().forEach((t) => t.stop());
      node.disconnect();
      src.disconnect();
      void ctx.close();
    },
  };
}

export interface PlayerEvents {
  onStarted(turnId: number): void;
  onDrained(turnId: number): void;
}

/**
 * Schedules PCM chunks back to back. Every chunk carries the turn id; chunks for a
 * turn that has been stopped are dropped, so stale audio can never play after a
 * barge-in, even if it was already in flight on the socket.
 */
export class Player {
  private ctx: AudioContext;
  private gain: GainNode;
  private sources = new Set<AudioBufferSourceNode>();
  private nextTime = 0;
  private stopped = new Set<number>();
  private maxStopped = 0;
  private startedTurns = new Set<number>();
  private loopback: { pc1: RTCPeerConnection; pc2: RTCPeerConnection; el: HTMLAudioElement } | null = null;

  constructor(private events: PlayerEvents) {
    this.ctx = new AudioContext({ latencyHint: "interactive" });
    this.gain = this.ctx.createGain();
    this.gain.connect(this.ctx.destination);
  }

  async resume(): Promise<void> {
    if (this.ctx.state !== "running") await this.ctx.resume();
  }

  /**
   * Experimental: route playback through a WebRTC loopback so Chrome's echo
   * canceller can "see" it. Not verified on the target machine; headphones remain
   * the reliable option.
   */
  async setAecLoopback(on: boolean): Promise<void> {
    if (on === !!this.loopback) return;
    this.gain.disconnect();
    if (!on && this.loopback) {
      this.loopback.pc1.close();
      this.loopback.pc2.close();
      this.loopback.el.srcObject = null;
      this.loopback = null;
      this.gain.connect(this.ctx.destination);
      return;
    }
    const dest = this.ctx.createMediaStreamDestination();
    this.gain.connect(dest);
    const pc1 = new RTCPeerConnection();
    const pc2 = new RTCPeerConnection();
    pc1.onicecandidate = (e) => e.candidate && pc2.addIceCandidate(e.candidate);
    pc2.onicecandidate = (e) => e.candidate && pc1.addIceCandidate(e.candidate);
    const el = new Audio();
    el.autoplay = true;
    pc2.ontrack = (e) => {
      el.srcObject = e.streams[0];
      void el.play();
    };
    dest.stream.getTracks().forEach((t) => pc1.addTrack(t, dest.stream));
    const offer = await pc1.createOffer();
    await pc1.setLocalDescription(offer);
    await pc2.setRemoteDescription(offer);
    const answer = await pc2.createAnswer();
    await pc2.setLocalDescription(answer);
    await pc1.setRemoteDescription(answer);
    this.loopback = { pc1, pc2, el };
  }

  enqueue(turnId: number, sampleRate: number, pcm: Int16Array): void {
    if (this.stopped.has(turnId) || turnId <= this.maxStopped) return;
    const buf = this.ctx.createBuffer(1, pcm.length, sampleRate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.gain);
    const now = this.ctx.currentTime;
    const startAt = Math.max(now + 0.02, this.nextTime);
    this.nextTime = startAt + buf.duration;
    src.start(startAt);
    this.sources.add(src);
    (src as unknown as { turnId: number }).turnId = turnId;
    if (!this.startedTurns.has(turnId)) {
      this.startedTurns.add(turnId);
      const delayMs = Math.max(0, (startAt - now) * 1000);
      setTimeout(() => {
        if (!this.stopped.has(turnId)) this.events.onStarted(turnId);
      }, delayMs);
    }
    src.onended = () => {
      this.sources.delete(src);
      const stillThisTurn = [...this.sources].some((s) => (s as unknown as { turnId: number }).turnId === turnId);
      if (!stillThisTurn && !this.stopped.has(turnId)) this.events.onDrained(turnId);
    };
  }

  /** Stop everything for ``turnId`` (and older) immediately. Returns ms spent. */
  stop(turnId: number): number {
    const t0 = performance.now();
    this.stopped.add(turnId);
    this.maxStopped = Math.max(this.maxStopped, turnId);
    for (const s of this.sources) {
      try {
        s.onended = null;
        s.stop();
        s.disconnect();
      } catch {
        /* already stopped */
      }
    }
    this.sources.clear();
    this.nextTime = 0;
    this.gain.gain.cancelScheduledValues(this.ctx.currentTime);
    this.gain.gain.value = 1;
    return performance.now() - t0;
  }

  duck(on: boolean): void {
    const g = this.gain.gain;
    g.cancelScheduledValues(this.ctx.currentTime);
    g.setTargetAtTime(on ? 0.15 : 1, this.ctx.currentTime, 0.02);
  }

  get playing(): boolean {
    return this.sources.size > 0;
  }

  close(): void {
    void this.setAecLoopback(false);
    void this.ctx.close();
  }
}

/** Parse a server audio message: 16-byte header (turn, seq, rate, sentence) + Int16 PCM. */
export function parseAudio(buf: ArrayBuffer): { turnId: number; seq: number; rate: number; idx: number; pcm: Int16Array } {
  const dv = new DataView(buf);
  return {
    turnId: dv.getUint32(0, true),
    seq: dv.getUint32(4, true),
    rate: dv.getUint32(8, true),
    idx: dv.getUint32(12, true),
    pcm: new Int16Array(buf.slice(16)),
  };
}
