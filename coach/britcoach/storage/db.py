"""Optional local session history in SQLite (``data/coach.sqlite3``).

Writes are tiny and run via ``asyncio.to_thread`` so they never block the loop.
Disable entirely with ``COACH_SAVE_HISTORY=false``.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, started_at REAL, ended_at REAL, mode TEXT, scenario TEXT, model TEXT,
  stats_json TEXT
);
CREATE TABLE IF NOT EXISTS turns (
  session_id TEXT, idx INTEGER, role TEXT, text TEXT, interrupted INTEGER, ts REAL,
  PRIMARY KEY (session_id, idx)
);
CREATE TABLE IF NOT EXISTS issues (
  session_id TEXT, turn_idx INTEGER, category TEXT, original TEXT, suggestion TEXT, severity TEXT
);
CREATE TABLE IF NOT EXISTS reviews (session_id TEXT PRIMARY KEY, created_at REAL, json TEXT);
CREATE TABLE IF NOT EXISTS metrics (session_id TEXT, ts REAL, json TEXT);
CREATE TABLE IF NOT EXISTS cloud_calls (
  ts REAL, session_id TEXT, provider TEXT, model TEXT, purpose TEXT, chars_sent INTEGER, ok INTEGER, reason TEXT
);
"""


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ sync core
    def _exec(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, args)
            self._conn.commit()

    def _query(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    # ------------------------------------------------------------------ async API
    async def start_session(self, sid: str, mode: str, scenario: str, model: str) -> None:
        await asyncio.to_thread(self._exec, "INSERT OR REPLACE INTO sessions (id, started_at, mode, scenario, model) VALUES (?,?,?,?,?)",
                                (sid, time.time(), mode, scenario, model))

    async def save_transcript(self, sid: str, transcript: list[dict], stats: dict) -> None:
        def work() -> None:
            with self._lock:
                c = self._conn
                c.execute("DELETE FROM turns WHERE session_id=?", (sid,))
                c.execute("DELETE FROM issues WHERE session_id=?", (sid,))
                for idx, t in enumerate(transcript):
                    c.execute("INSERT INTO turns VALUES (?,?,?,?,?,?)",
                              (sid, idx, t["role"], t["text"], int(t.get("interrupted", False)), t.get("ts", 0)))
                    for i in t.get("issues", []):
                        c.execute("INSERT INTO issues VALUES (?,?,?,?,?,?)",
                                  (sid, idx, i["category"], i["original"], i["suggestion"], i["severity"]))
                c.execute("UPDATE sessions SET ended_at=?, stats_json=? WHERE id=?", (time.time(), json.dumps(stats), sid))
                c.commit()

        await asyncio.to_thread(work)

    async def save_review(self, sid: str, review: dict) -> None:
        await asyncio.to_thread(self._exec, "INSERT OR REPLACE INTO reviews VALUES (?,?,?)", (sid, time.time(), json.dumps(review)))

    async def save_metrics(self, sid: str, metrics: dict) -> None:
        await asyncio.to_thread(self._exec, "INSERT INTO metrics VALUES (?,?,?)", (sid, time.time(), json.dumps(metrics)))

    async def log_cloud_call(self, sid: str, info: dict) -> None:
        await asyncio.to_thread(self._exec, "INSERT INTO cloud_calls VALUES (?,?,?,?,?,?,?,?)",
                                (time.time(), sid, info.get("provider"), info.get("model"), info.get("purpose"),
                                 info.get("chars_sent", 0), int(info.get("ok", False)), info.get("reason", "")))

    async def list_sessions(self, limit: int = 50) -> list[dict]:
        rows = await asyncio.to_thread(self._query, "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["stats"] = json.loads(r.pop("stats_json") or "{}")
        return rows

    async def get_session(self, sid: str) -> dict | None:
        rows = await asyncio.to_thread(self._query, "SELECT * FROM sessions WHERE id=?", (sid,))
        if not rows:
            return None
        s = rows[0]
        s["stats"] = json.loads(s.pop("stats_json") or "{}")
        s["turns"] = await asyncio.to_thread(self._query, "SELECT * FROM turns WHERE session_id=? ORDER BY idx", (sid,))
        rev = await asyncio.to_thread(self._query, "SELECT json FROM reviews WHERE session_id=?", (sid,))
        s["review"] = json.loads(rev[0]["json"]) if rev else None
        return s

    async def progress(self, limit: int = 20) -> dict:
        """Per-session error rate and top categories, oldest first, for the progress chart."""
        sessions = list(reversed(await self.list_sessions(limit)))
        cats = await asyncio.to_thread(self._query,
                                       "SELECT category, COUNT(*) AS n FROM issues GROUP BY category ORDER BY n DESC LIMIT 6")
        points = [{"session_id": s["id"], "started_at": s["started_at"], "mode": s["mode"],
                   "issues_per_100_words": s["stats"].get("issues_per_100_words"),
                   "user_words": s["stats"].get("user_words"), "level": s["stats"].get("estimated_level")}
                  for s in sessions if s["stats"]]
        return {"sessions": points, "top_categories": cats}

    async def cloud_calls(self, limit: int = 50) -> list[dict]:
        return await asyncio.to_thread(self._query, "SELECT * FROM cloud_calls ORDER BY ts DESC LIMIT ?", (limit,))
