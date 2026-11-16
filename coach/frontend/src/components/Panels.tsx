import type { ChatItem, Metrics, Options, SessionConfig } from "../lib/types";

const STATE_LABEL: Record<string, string> = {
  idle: "Idle",
  listening: "Listening",
  user_speaking: "You're speaking",
  thinking: "Thinking",
  speaking: "Speaking",
};

export function StatusPill({ state, ducked }: { state: string; ducked: boolean }) {
  return (
    <span className={`pill pill-${state}`} data-testid="state">
      <span className="dot" /> {STATE_LABEL[state] ?? state}
      {ducked ? " · checking for echo" : ""}
    </span>
  );
}

export function MicMeter({ level }: { level: number }) {
  const pct = Math.min(100, Math.round(Math.sqrt(level) * 220));
  return (
    <div className="meter" title="Microphone level" aria-label="Microphone level">
      <div style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Transcript({ chat, partial }: { chat: ChatItem[]; partial: string }) {
  return (
    <div className="transcript" data-testid="transcript">
      {chat.length === 0 && <p className="muted">Press Start and say hello. The coach will open the conversation.</p>}
      {chat.map((m) => (
        <div key={m.id} className={`bubble ${m.role}${m.interrupted ? " interrupted" : ""}`}>
          <div className="who">{m.role === "user" ? "You" : "Coach"}</div>
          <div className="text">
            {m.interrupted && m.heard !== undefined ? (
              <>
                {m.heard}
                <span className="cut">{m.text.slice(m.heard.length)}</span>
                <span className="tag">interrupted</span>
              </>
            ) : (
              m.text
            )}
          </div>
          {m.issues && m.issues.length > 0 && (
            <div className="issues">
              {m.issues.map((i, k) => (
                <span key={k} className={`chip ${i.severity}`} title={i.explanation}>
                  <s>{i.original}</s> → {i.suggestion}
                  {i.source === "cloud" ? " ☁" : ""}
                </span>
              ))}
              {m.strategy && m.strategy !== "none" && <span className="strategy">{m.strategy === "brief" ? "quick tip" : "recast"}</span>}
            </div>
          )}
        </div>
      ))}
      {partial && <div className="bubble user partial">{partial}…</div>}
    </div>
  );
}

interface SettingsProps {
  options: Options | null;
  config: SessionConfig | null;
  onChange: (c: Partial<SessionConfig>) => void;
  onSelectModel: (backend: string, profile: string) => void;
  modelBusy: boolean;
  aecLoopback: boolean;
  setAecLoopback: (v: boolean) => void;
  cloudKeys: { anthropic: boolean; openai: boolean };
}

export function SettingsPanel({ options, config, onChange, onSelectModel, modelBusy, aecLoopback, setAecLoopback, cloudKeys }: SettingsProps) {
  if (!options || !config) return <div className="card">Loading settings…</div>;
  const scenarios = options.scenarios.filter((s) => s.modes.includes(config.mode));
  const cur = options.current_llm || {};
  const profiles = Object.entries(options.llm_profiles);
  const backends = options.llm_backends.filter((b) => b !== "mlx" || options.mlx_supported);
  return (
    <div className="card settings">
      <h3>Conversation</h3>
      <label>
        Mode
        <select value={config.mode} onChange={(e) => onChange({ mode: e.target.value })} data-testid="mode">
          {options.modes.map((m) => (
            <option key={m.id} value={m.id}>{m.title}</option>
          ))}
        </select>
      </label>
      <p className="hint">{options.modes.find((m) => m.id === config.mode)?.description}</p>
      <label>
        Scenario
        <select value={scenarios.some((s) => s.id === config.scenario) ? config.scenario : ""} onChange={(e) => onChange({ scenario: e.target.value })}>
          {!scenarios.some((s) => s.id === config.scenario) && <option value="">Choose…</option>}
          {scenarios.map((s) => (
            <option key={s.id} value={s.id}>{s.title}</option>
          ))}
        </select>
      </label>
      {config.mode === "roleplay" && (
        <label>
          Difficulty
          <select value={config.difficulty} onChange={(e) => onChange({ difficulty: e.target.value })}>
            {options.difficulties.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>
      )}
      <label>
        Level
        <select value={config.level} onChange={(e) => onChange({ level: e.target.value })}>
          <option value="auto">Adapt automatically</option>
          <option value="B1">B1</option>
          <option value="B2">B2</option>
          <option value="C1">C1</option>
        </select>
      </label>
      <label>
        Corrections
        <select value={config.sensitivity} onChange={(e) => onChange({ sensitivity: e.target.value })} data-testid="sensitivity">
          <option value="off">Off (review only)</option>
          <option value="low">Low: important mistakes, recast only</option>
          <option value="medium">Medium: recast, quick tip if repeated</option>
          <option value="high">High: quick tips for important mistakes</option>
        </select>
      </label>

      <h3>Voice &amp; model</h3>
      <label>
        British voice
        <select value={config.voice} onChange={(e) => onChange({ voice: e.target.value })}>
          {options.voices.map((v) => (
            <option key={v.id} value={v.id} disabled={!v.installed}>
              {v.label}{v.installed ? "" : " (not installed)"}
            </option>
          ))}
        </select>
      </label>
      <label>
        Model
        <select
          disabled={modelBusy}
          value=""
          onChange={(e) => {
            const [b, p] = e.target.value.split(":");
            if (b && p) onSelectModel(b, p);
          }}
        >
          <option value="">{String(cur.backend ?? "?")} · {String(cur.model ?? "none")}{modelBusy ? " (loading…)" : ""}</option>
          {backends.filter((b) => b !== "mock").flatMap((b) =>
            profiles.map(([id, p]) => (
              <option key={`${b}:${id}`} value={`${b}:${id}`}>{b} · {p.label}</option>
            )),
          )}
        </select>
      </label>

      <h3>Turn-taking</h3>
      <label>
        Interruptions (barge-in)
        <select value={config.barge_in_mode} onChange={(e) => onChange({ barge_in_mode: e.target.value })}>
          <option value="headphones">Headphones: interrupt instantly</option>
          <option value="speakers">Laptop speakers: echo-checked</option>
          <option value="off">Off: don't listen while the coach speaks</option>
        </select>
      </label>
      {config.barge_in_mode === "speakers" && (
        <label className="check">
          <input type="checkbox" checked={aecLoopback} onChange={(e) => setAecLoopback(e.target.checked)} />
          Experimental: route playback via WebRTC so Chrome's echo canceller sees it
        </label>
      )}
      <label>
        Pause before the coach replies: {config.end_of_turn_ms} ms
        <input type="range" min={300} max={1500} step={20} value={config.end_of_turn_ms}
               onChange={(e) => onChange({ end_of_turn_ms: Number(e.target.value) })} />
      </label>

      <h3>Cloud (optional)</h3>
      <label className="check">
        <input type="checkbox" checked={config.cloud_enabled} onChange={(e) => onChange({ cloud_enabled: e.target.checked })} data-testid="cloud" />
        Allow cloud analysis
      </label>
      {config.cloud_enabled && (
        <>
          <label>
            Provider
            <select value={config.cloud_provider} onChange={(e) => onChange({ cloud_provider: e.target.value })}>
              <option value="anthropic">Anthropic{cloudKeys.anthropic ? "" : " (no ANTHROPIC_API_KEY)"}</option>
              <option value="openai">OpenAI{cloudKeys.openai ? "" : " (no OPENAI_API_KEY)"}</option>
            </select>
          </label>
          <label className="check">
            <input type="checkbox" checked={config.cloud_turn_corrections} onChange={(e) => onChange({ cloud_turn_corrections: e.target.checked })} />
            Cloud-assisted corrections (sends each of your sentences)
          </label>
          <p className="hint warn">Conversation always stays local. Text is only sent for the features ticked here or when you ask for an advanced review.</p>
        </>
      )}
    </div>
  );
}

const ROWS: [string, string][] = [
  ["eot_ms", "End-of-turn detection"],
  ["stt_wait_ms", "STT (on critical path)"],
  ["stt_compute_ms", "STT compute"],
  ["llm_ttft_ms", "LLM time to first token"],
  ["llm_tokens_per_s", "LLM tokens/s"],
  ["first_sentence_ms", "First speakable chunk"],
  ["tts_first_audio_ms", "TTS first audio"],
  ["e2e_server_ms", "Speech end → first audio sent"],
  ["e2e_audible_ms", "Speech end → audible"],
  ["bargein_detect_ms", "Barge-in detection"],
  ["bargein_cancel_ms", "Barge-in cancel (server)"],
  ["bargein_client_stop_ms", "Barge-in stop (browser)"],
  ["vad_frame_ms", "VAD per 32 ms frame"],
  ["rss_mb", "Server RSS (MB)"],
  ["sys_available_gb", "System free memory (GB)"],
];

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function LatencyPanel({ metrics, lastBargeIn }: { metrics: Metrics[]; lastBargeIn: Metrics | null }) {
  const userTurns = metrics.filter((m) => m.kind === "user");
  const last = userTurns[userTurns.length - 1] || metrics[metrics.length - 1];
  const fmt = (v: unknown) => (typeof v === "number" ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)) : "–");
  return (
    <div className="card latency" data-testid="latency">
      <h3>Latency diagnostics</h3>
      {!last && !lastBargeIn && <p className="muted">Measured live from your own turns. Nothing yet.</p>}
      {(last || lastBargeIn) && (
        <table>
          <thead>
            <tr><th>Stage</th><th>Last</th><th>Median</th></tr>
          </thead>
          <tbody>
            {ROWS.map(([k, label]) => {
              const lastV = last?.[k] ?? lastBargeIn?.[k];
              const med = median(userTurns.map((m) => m[k]).filter((v): v is number => typeof v === "number"));
              if (lastV === undefined && med === null) return null;
              const target = k === "e2e_audible_ms" && typeof lastV === "number";
              return (
                <tr key={k} className={target ? (lastV <= 1000 ? "good" : "slow") : ""}>
                  <td>{label}</td><td>{fmt(lastV)}</td><td>{fmt(med)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <p className="hint">Target ≈ 1000 ms from the end of your speech to audible reply. That's a goal, not a guarantee. End-of-turn detection deliberately waits for the pause you set above.</p>
    </div>
  );
}
