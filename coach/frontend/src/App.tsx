import { useEffect, useState } from "react";
import { LatencyPanel, MicMeter, SettingsPanel, StatusPill, Transcript } from "./components/Panels";
import { ReviewView } from "./components/ReviewView";
import type { Options } from "./lib/types";
import { useCoach } from "./lib/useCoach";

export default function App() {
  const c = useCoach();
  const [options, setOptions] = useState<Options | null>(null);
  const [typed, setTyped] = useState("");
  const [view, setView] = useState<"chat" | "review">("chat");
  const [waitingReview, setWaitingReview] = useState(false);
  const [modelBusy, setModelBusy] = useState(false);
  const [aecLoopback, setAecLoopback] = useState(false);
  const [health, setHealth] = useState<{ status: string; problems: string } | null>(null);

  const loadOptions = () => fetch("/api/options").then((r) => r.json()).then(setOptions).catch(() => undefined);
  useEffect(() => {
    void loadOptions();
    fetch("/api/health").then((r) => r.json()).then(setHealth).catch(() => undefined);
  }, []);
  useEffect(() => {
    if (c.review) setWaitingReview(false);
  }, [c.review]);
  useEffect(() => {
    void c.player.current?.setAecLoopback(aecLoopback);
  }, [aecLoopback, c.player]);

  const selectModel = async (backend: string, profile: string) => {
    setModelBusy(true);
    try {
      const r = await fetch("/api/llm/select", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ backend, profile }) });
      if (!r.ok) alert((await r.json()).detail);
      await loadOptions();
    } finally {
      setModelBusy(false);
    }
  };

  const requestReview = (opts: { enrich?: boolean; use_cloud?: boolean; consent?: boolean }) => {
    setWaitingReview(true);
    c.setReview(null);
    setView("review");
    c.endSession(opts);
  };

  const s = (options?.settings || {}) as Record<string, unknown>;
  const cloudKeys = { anthropic: !!s.anthropic_key_present, openai: !!s.openai_key_present };

  return (
    <div className="app">
      <header>
        <div>
          <h1>British Conversation Coach</h1>
          <span className="muted small">Local voice practice · {String(options?.current_llm?.model ?? "")}</span>
        </div>
        <div className="head-right">
          {c.cloudActive && <span className="pill cloud" data-testid="cloud-active">☁ Cloud: {String(c.cloudActive.provider)} ({String(c.cloudActive.purpose)})</span>}
          {!c.cloudActive && <span className="pill local">● Local only</span>}
          <StatusPill state={c.state} ducked={c.ducked} />
        </div>
      </header>

      {health && health.status !== "ok" && <div className="banner">Some engines aren't ready: {health.problems}</div>}
      {c.errors.map((e, i) => <div key={i} className="banner">{e}</div>)}

      {view === "review" ? (
        <ReviewView review={c.review} waiting={waitingReview} cloudEnabled={!!c.config?.cloud_enabled}
                    cloudProvider={c.config?.cloud_provider ?? "anthropic"} onRequest={requestReview}
                    onClose={() => { c.newSession(); setView("chat"); }} />
      ) : (
        <main>
          <section className="conversation">
            <div className="controls">
              {!c.running ? (
                <button className="primary" data-testid="start" onClick={() => void c.start(c.config ?? {})}>Start conversation</button>
              ) : (
                <button className="danger" data-testid="stop" onClick={c.stop}>Stop</button>
              )}
              <button className="ghost" onClick={() => requestReview({ enrich: true })} disabled={c.chat.length === 0}>End &amp; review</button>
              <MicMeter level={c.level} />
            </div>
            <Transcript chat={c.chat} partial={c.partial} />
            <form className="typed" onSubmit={(e) => { e.preventDefault(); if (typed.trim()) { c.sendText(typed.trim()); setTyped(""); } }}>
              <input value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="…or type a message" aria-label="Type a message" />
              <button type="submit" disabled={!c.running}>Send</button>
            </form>
            <p className="hint">Tip: use headphones for instant interruptions. With laptop speakers, choose "Laptop speakers" so the coach doesn't interrupt itself.</p>
          </section>
          <aside>
            <SettingsPanel options={options} config={c.config} onChange={c.configure} onSelectModel={selectModel}
                           modelBusy={modelBusy} aecLoopback={aecLoopback} setAecLoopback={setAecLoopback} cloudKeys={cloudKeys} />
            <LatencyPanel metrics={c.metrics} lastBargeIn={c.lastBargeIn} />
            {c.cloudLog.length > 0 && (
              <div className="card">
                <h3>Cloud calls this session</h3>
                <ul className="small">
                  {c.cloudLog.map((l, i) => <li key={i}>{String(l.provider)} · {String(l.purpose)} · {String(l.chars_sent)} chars · {l.ok ? "ok" : "failed"}</li>)}
                </ul>
              </div>
            )}
          </aside>
        </main>
      )}
    </div>
  );
}
