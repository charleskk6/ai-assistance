export type CoachState = "idle" | "listening" | "user_speaking" | "thinking" | "speaking";

export interface Issue {
  category: string;
  original: string;
  suggestion: string;
  explanation: string;
  severity: string;
  source: string;
}

export interface ChatItem {
  id: string;
  role: "user" | "assistant";
  text: string;
  turnId?: number;
  issues?: Issue[];
  strategy?: string;
  interrupted?: boolean;
  heard?: string;
  dropped?: string; // cut off by the length limit and not spoken
  pending?: boolean;
}

export interface SessionConfig {
  mode: string;
  scenario: string;
  difficulty: string;
  level: string;
  sensitivity: string;
  voice: string;
  barge_in_mode: string;
  end_of_turn_ms: number;
  cloud_enabled: boolean;
  cloud_provider: string;
  cloud_turn_corrections: boolean;
  session_id?: string;
}

export interface Options {
  modes: { id: string; title: string; description: string; sensitivity: string }[];
  scenarios: { id: string; title: string; modes: string[] }[];
  difficulties: string[];
  llm_profiles: Record<string, { label: string; licence: string; approx_ram_gb: number; [k: string]: unknown }>;
  llm_backends: string[];
  mlx_supported: boolean;
  current_llm: Record<string, unknown> | null;
  voices: { id: string; label: string; installed: boolean; backend: string }[];
  default_voice: string;
  tts_backend: string | null;
  settings: Record<string, unknown>;
  platform: { system: string; machine: string; python: string };
}

export type Metrics = Record<string, number | boolean | string | null | undefined>;

export interface Review {
  session_id: string;
  mistakes: { you_said: string; better: string; explanation: string; category: string; times: number; in_sentence: string; source: string }[];
  expressions: { phrase: string; meaning: string; example: string; level: string; heard_today: boolean }[];
  alternatives: { original: string; natural: string; source: string }[];
  practice: string[];
  stats: Record<string, unknown>;
  notes: string[];
  enriched_by: string | null;
  rejected_llm_items: number;
  transcript: { role: string; text: string; interrupted: boolean }[];
  metrics: Metrics[];
}
