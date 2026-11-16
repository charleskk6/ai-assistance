import { useEffect, useState } from "react";
import type { Review } from "../lib/types";

interface Props {
  review: Review | null;
  waiting: boolean;
  cloudEnabled: boolean;
  cloudProvider: string;
  onRequest: (opts: { enrich?: boolean; use_cloud?: boolean; consent?: boolean }) => void;
  onClose: () => void;
}

interface Progress {
  sessions: { session_id: string; started_at: number; issues_per_100_words: number | null; user_words: number | null; level: string | null }[];
  top_categories: { category: string; n: number }[];
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return <span className="muted">More sessions needed for a trend.</span>;
  const w = 220, h = 48, max = Math.max(...values, 1);
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * w},${h - (v / max) * (h - 6) - 3}`).join(" ");
  return (
    <svg width={w} height={h} className="spark" role="img" aria-label="Issues per 100 words over sessions">
      <polyline points={pts} fill="none" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

export function ReviewView({ review, waiting, cloudEnabled, cloudProvider, onRequest, onClose }: Props) {
  const [consent, setConsent] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  useEffect(() => {
    fetch("/api/progress").then((r) => r.json()).then(setProgress).catch(() => undefined);
  }, [review]);

  return (
    <div className="review" data-testid="review">
      <div className="review-head">
        <h2>Session review</h2>
        <button className="ghost" onClick={onClose}>Back to conversation</button>
      </div>
      {waiting && !review && <p className="muted">Analysing your session…</p>}
      {review && (
        <>
          <section>
            <h3>Three things to work on</h3>
            {review.mistakes.length === 0 && <p>No clear grammar or word-choice mistakes were found in what was transcribed. Nice one!</p>}
            <ol>
              {review.mistakes.map((m, i) => (
                <li key={i}>
                  <div><s>{m.you_said}</s> → <strong>{m.better}</strong> {m.times > 1 && <span className="tag">×{m.times}</span>}</div>
                  <div className="muted">{m.explanation}</div>
                  <div className="quote">You said: “{m.in_sentence}”</div>
                </li>
              ))}
            </ol>
          </section>
          <section>
            <h3>Five useful British expressions</h3>
            <ul>
              {review.expressions.map((e, i) => (
                <li key={i}>
                  <strong>{e.phrase}</strong> <span className="tag">{e.level}</span>{e.heard_today && <span className="tag heard">heard today</span>}
                  <div className="muted">{/[.!?]$/.test(e.meaning) ? e.meaning : `${e.meaning}.`} <em>“{e.example}”</em></div>
                </li>
              ))}
            </ul>
          </section>
          {review.alternatives.length > 0 && (
            <section>
              <h3>More natural ways to say it</h3>
              <ul>
                {review.alternatives.map((a, i) => (
                  <li key={i}>
                    <div className="muted">“{a.original}”</div>
                    <div>→ “{a.natural}” <span className="tag">{a.source}</span></div>
                  </li>
                ))}
              </ul>
            </section>
          )}
          <section>
            <h3>Practice suggestions</h3>
            <ul>{review.practice.map((p, i) => <li key={i}>{p}</li>)}</ul>
          </section>
          <section className="stats">
            <h3>This session</h3>
            <div className="grid">
              {Object.entries(review.stats).filter(([, v]) => typeof v !== "object").map(([k, v]) => (
                <div key={k}><div className="muted">{k.replace(/_/g, " ")}</div><div className="big">{String(v)}</div></div>
              ))}
            </div>
            {review.notes.map((n, i) => <p key={i} className="hint">{n}</p>)}
            {review.enriched_by && <p className="hint">Extra findings from: {review.enriched_by}{review.rejected_llm_items ? ` (${review.rejected_llm_items} unsupported item(s) discarded)` : ""}</p>}
          </section>
          <section>
            <h3>Deeper analysis</h3>
            <div className="row">
              <button onClick={() => onRequest({ enrich: true })}>Re-analyse with the local model</button>
            </div>
            {cloudEnabled ? (
              <div className="cloud-consent">
                <label className="check">
                  <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
                  Send what I said in this session ({review.stats.user_words as number} words) to {cloudProvider} for advanced analysis
                </label>
                <button disabled={!consent} onClick={() => onRequest({ enrich: false, use_cloud: true, consent: true })}>
                  Advanced analysis (cloud)
                </button>
              </div>
            ) : (
              <p className="hint">Cloud analysis is off. Enable it in settings if you want an advanced review.</p>
            )}
          </section>
        </>
      )}
      <section>
        <h3>Progress</h3>
        {progress && (
          <>
            <div className="row">
              <div>
                <div className="muted">Issues per 100 words, by session</div>
                <Sparkline values={progress.sessions.map((s) => s.issues_per_100_words ?? 0)} />
              </div>
              <div>
                <div className="muted">Most frequent areas</div>
                {progress.top_categories.map((c) => <span key={c.category} className="chip">{c.category} · {c.n}</span>)}
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
