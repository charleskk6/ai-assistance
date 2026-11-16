import { useCallback, useEffect, useRef, useState } from "react";
import { Player, parseAudio, startCapture, type CaptureHandle } from "./audio";
import type { ChatItem, CoachState, Metrics, Review, SessionConfig } from "./types";

type Msg = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

let uid = 0;
const nextId = () => `m${++uid}`;

export function useCoach() {
  const wsRef = useRef<WebSocket | null>(null);
  const playerRef = useRef<Player | null>(null);
  const captureRef = useRef<CaptureHandle | null>(null);
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<CoachState>("idle");
  const [running, setRunning] = useState(false);
  const [chat, setChat] = useState<ChatItem[]>([]);
  const [partial, setPartial] = useState("");
  const [config, setConfig] = useState<SessionConfig | null>(null);
  const [metrics, setMetrics] = useState<Metrics[]>([]);
  const [lastBargeIn, setLastBargeIn] = useState<Metrics | null>(null);
  const [level, setLevel] = useState(0);
  const [review, setReview] = useState<Review | null>(null);
  const [cloudActive, setCloudActive] = useState<Msg | null>(null);
  const [cloudLog, setCloudLog] = useState<Msg[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [engines, setEngines] = useState<Record<string, Msg>>({});
  const [ducked, setDucked] = useState(false);

  const send = useCallback((m: Msg) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(m));
  }, []);

  const onJson = useCallback(
    (m: Msg) => {
      switch (m.type) {
        case "hello":
          setConfig(m.config);
          setEngines(m.engines || {});
          break;
        case "config":
          setConfig(m.config);
          break;
        case "state":
          setState(m.state);
          break;
        case "partial_transcript":
          setPartial(m.text);
          break;
        case "user_final":
          setPartial("");
          setChat((c) => [
            ...c,
            { id: nextId(), role: "user", text: m.text, turnId: m.turn_id, issues: m.issues, strategy: m.correction?.strategy },
          ]);
          break;
        case "assistant_delta":
          setChat((c) => {
            const last = c[c.length - 1];
            if (last && last.role === "assistant" && last.turnId === m.turn_id) {
              return [...c.slice(0, -1), { ...last, text: last.text + m.text }];
            }
            return [...c, { id: nextId(), role: "assistant", text: m.text, turnId: m.turn_id, pending: true }];
          });
          break;
        case "assistant_done":
          setChat((c) => c.map((x) => (x.role === "assistant" && x.turnId === m.turn_id ? { ...x, pending: false } : x)));
          break;
        case "stop_audio": {
          const ms = playerRef.current?.stop(m.turn_id) ?? 0;
          send({ type: "playback", event: "stopped", turn_id: m.turn_id, stop_ms: Math.round(ms * 100) / 100 });
          setDucked(false);
          break;
        }
        case "assistant_interrupted":
          setChat((c) =>
            c.map((x) =>
              x.role === "assistant" && x.turnId === m.turn_id ? { ...x, interrupted: true, heard: m.heard, pending: false } : x,
            ),
          );
          break;
        case "assistant_truncated":
          setChat((c) => c.map((x) => (x.role === "assistant" && x.turnId === m.turn_id ? { ...x, dropped: m.dropped } : x)));
          break;
        case "barge_in":
          setLastBargeIn(m.metrics);
          break;
        case "duck":
          playerRef.current?.duck(!!m.on);
          setDucked(!!m.on);
          break;
        case "metrics":
          setMetrics((ms) => [...ms.slice(-49), m.turn]);
          break;
        case "metric_update":
          setLastBargeIn((b) => ({ ...(b || {}), ...m }));
          break;
        case "review":
          setReview(m.review);
          break;
        case "cloud_status":
          if (m.active) setCloudActive(m);
          else {
            setCloudActive(null);
            setCloudLog((l) => [...l.slice(-19), m]);
          }
          break;
        case "cloud_correction":
          setChat((c) => c.map((x) => (x.role === "user" && x.text === m.text ? { ...x, issues: [...(x.issues || []), ...m.issues] } : x)));
          break;
        case "echo_suspected":
          setErrors((e) => [
            ...e.slice(-4),
            `The coach seems to have heard its own voice ("${m.heard}") and stopped. Interruptions are now set to ` +
              `"Laptop speakers" (echo-checked). Use headphones for the snappiest conversation.`,
          ]);
          break;
        case "cloud_error":
        case "error":
          setErrors((e) => [...e.slice(-4), m.message]);
          break;
      }
    },
    [send],
  );

  const connect = useCallback((): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return resolve();
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        setConnected(true);
        resolve();
      };
      ws.onerror = () => reject(new Error("WebSocket error"));
      ws.onclose = () => {
        setConnected(false);
        setRunning(false);
        setState("idle");
      };
      ws.onmessage = (e) => {
        if (e.data instanceof ArrayBuffer) {
          const a = parseAudio(e.data);
          playerRef.current?.enqueue(a.turnId, a.rate, a.pcm);
        } else onJson(JSON.parse(e.data));
      };
      wsRef.current = ws;
    });
  }, [onJson]);

  useEffect(() => {
    void connect().catch(() => undefined);
    return () => {
      captureRef.current?.stop();
      playerRef.current?.close();
      wsRef.current?.close();
    };
  }, [connect]);

  const start = useCallback(
    async (cfg: Partial<SessionConfig>) => {
      setErrors([]);
      setReview(null);
      await connect();
      if (!playerRef.current) {
        playerRef.current = new Player({
          onStarted: (turnId) => send({ type: "playback", event: "started", turn_id: turnId }),
          onDrained: (turnId) => send({ type: "playback", event: "ended", turn_id: turnId }),
        });
      }
      await playerRef.current.resume();
      try {
        captureRef.current = await startCapture(
          (pcm) => {
            const ws = wsRef.current;
            if (ws && ws.readyState === WebSocket.OPEN) ws.send(pcm);
          },
          (rms) => setLevel(rms),
        );
      } catch (e) {
        setErrors((x) => [...x, `Microphone unavailable: ${(e as Error).message}. You can still type.`]);
      }
      send({ type: "start", config: cfg });
      setRunning(true);
    },
    [connect, send],
  );

  const stop = useCallback(() => {
    captureRef.current?.stop();
    captureRef.current = null;
    send({ type: "stop" });
    setRunning(false);
    setLevel(0);
  }, [send]);

  const endSession = useCallback(
    (opts: { enrich?: boolean; use_cloud?: boolean; consent?: boolean }) => {
      captureRef.current?.stop();
      captureRef.current = null;
      setRunning(false);
      send({ type: "end_session", ...opts });
    },
    [send],
  );

  const newSession = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setChat([]);
    setMetrics([]);
    setReview(null);
    setPartial("");
    void connect();
  }, [connect]);

  return {
    connected, state, running, chat, partial, config, metrics, lastBargeIn, level, review, cloudActive, cloudLog,
    errors, engines, ducked, player: playerRef,
    start, stop, endSession, newSession,
    configure: (c: Partial<SessionConfig>) => send({ type: "config", config: c }),
    sendText: (text: string) => send({ type: "text_input", text }),
    setReview,
  };
}
