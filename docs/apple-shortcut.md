# The iPhone Shortcut: Siri → MacBook → Apple Cantonese voice

This builds the Shortcut that makes the whole thing work. Roughly ten minutes.

Before you start, the backend must be running on the Mac and reachable from the
iPhone — see [`local-network.md`](local-network.md) and confirm this works **on
the iPhone, in Safari**:

```
http://<your-mac>.local:8000/health
```

If that returns JSON in Safari, the Shortcut will work. If it doesn't, fix that
first — nothing below can compensate for an unreachable Mac.

---

## Build the Shortcut

Open **Shortcuts** on the iPhone → **+** (top right).

### 1. Name it — this is what you say to Siri

Tap the name at the top → **Rename** → type:

```
Ask My Assistant
```

The name *is* the Siri phrase. You will say "Hey Siri, **Ask My Assistant**".
Pick something Siri hears reliably; short English names work best even if you
then speak Cantonese. Avoid names that collide with app names.

### 2. Dictate Text

Search the action list for **Dictate Text** → add it.

Tap the arrow to expand it and set:

| Setting | Value |
|---|---|
| Language | **Cantonese (Hong Kong)** — 廣東話（香港） |
| Stop Listening | **After Pause** |

Setting the language explicitly matters. On "Automatic" the dictation often
falls back to your system language and mangles Cantonese.

#### Mixing Cantonese and English

iOS dictates **one language at a time** - there is no mixed mode. Cantonese
(Hong Kong) is nonetheless the right setting, because Apple's Cantonese
recogniser is trained on Hong Kong speech, which is natively code-switched.
Common terms (`file`, `download`, `server`, `update`, `Python`) come through
fine.

Multi-word technical jargon is where it breaks down - "dependency injection",
"vector database" often arrive as Chinese homophones. Three things help:

- A slight **pause before and after** the English phrase, so the recogniser
  segments it.
- **Enable the English keyboard** (Settings -> General -> Keyboard) even if you
  never type with it; the English acoustic model stays loaded.
- For English-heavy questions, **duplicate this Shortcut** with `Language:
  English` and a different name ("Ask My Assistant in English"). Two Shortcuts,
  one backend, no code change.

The backend also tells the model that `source: siri` input is dictated and may
contain garbled English, so it infers the intended term rather than answering
the homophone. That handles the common cases; it cannot rescue a transcription
that lost the meaning entirely.

### 3. Get Contents of URL

Add **Get Contents of URL**. Put the URL in the field:

```
http://your-mac.local:8000/ask
```

(Replace `your-mac` with your Mac's actual local hostname.)

Expand the action with the arrow and set:

- **Method**: `POST`
- **Headers** — tap *Add new header* twice:

  | Key | Value |
  |---|---|
  | `Authorization` | `Bearer YOUR_TOKEN_HERE` |
  | `Content-Type` | `application/json` |

  `YOUR_TOKEN_HERE` is the exact value of `LOCAL_ASSISTANT_TOKEN` in your
  `.env` on the Mac. The word `Bearer`, one space, then the token.

- **Request Body**: `JSON`, then add three fields:

  | Key | Type | Value |
  |---|---|---|
  | `query` | Text | *(the **Dictated Text** variable — see below)* |
  | `mode` | Text | `auto` |
  | `source` | Text | `siri` |

  For `query`, tap the value field, then tap the **variable button** just above
  the keyboard (or the `Shortcut Input`/variable chip in the suggestion bar) and
  choose **Dictated Text**. It must be the blue variable chip, not the literal
  words "Dictated Text".

`source: siri` is what switches the backend into speech mode: short, plain
sentences, no Markdown, no URLs read aloud.

### 4. Get Dictionary Value

Add **Get Dictionary Value**. Set:

- **Get**: `Value`
- **for**: `answer`
- **in**: `Contents of URL` (should be filled in automatically)

### 5. Speak Text

Add **Speak Text**. It should already reference the **Dictionary Value**; if it
doesn't, tap the field and pick it.

Expand the action with the arrow and set:

| Setting | Value |
|---|---|
| Voice | a **Cantonese** voice — e.g. **Sinji (粵語)** |
| Language | **Cantonese (Hong Kong)** |
| Rate | ~0.5 (adjust to taste) |
| Wait Until Finished | On |

If no Cantonese voice is listed: **Settings → Accessibility → Spoken Content →
Voices → Chinese → Cantonese (Hong Kong)** and download one. Sinji is the
standard Siri Cantonese voice; the "Enhanced"/"Premium" download sounds
noticeably better and is worth the space.

### 6. Turn off "Show When Run" (optional but recommended)

In the Shortcut's settings (the ⓘ / sliders icon), turn **Show When Run** off so
it doesn't open the Shortcuts app UI every time. Also enable **Pin to Start
Menu** if you want it on the Home Screen.

---

## Use it

> "Hey Siri, Ask My Assistant."

Siri opens the Shortcut, the dictation prompt appears, you speak:

> 「解釋下 dependency injection。」

A few seconds later the Cantonese voice reads the answer back.

Try both acceptance tests:

| You say | Expected route | Why |
|---|---|---|
| 解釋下 dependency injection | `local` | no current information needed |
| 而家最新 stable Python version 係邊個？ | `web` | 最新 + version triggers a web search |

You can force a route by starting the sentence with `/local` or `/web`, though
that is awkward to dictate — it is mostly for testing with `curl`.

---

## Optional: also show the answer on screen

Insert a **Show Result** action with the Dictionary Value before **Speak Text**.
Useful while debugging, since you can read what came back.

## Optional: speak the sources too

The backend returns `sources` separately, deliberately, because reading URLs
aloud is miserable. If you want the domains spoken, add a second
**Get Dictionary Value** for `sources`, then **Repeat with Each**, then
**Get Dictionary Value** for `domain` inside the loop. I would not bother.

---

## Troubleshooting

Work top to bottom — the causes are ordered by how often they are the problem.

### "Could not connect to the server" / the Shortcut hangs then fails

1. **Is the backend running?** On the Mac: `curl http://127.0.0.1:8000/health`.
   Nothing back → start it with `./run.sh`.
2. **Is it listening on the LAN, not just localhost?** On the Mac:
   ```
   lsof -nP -iTCP:8000 | grep LISTEN
   ```
   You want `*:8000`. If you see `127.0.0.1:8000`, the Mac is refusing the
   iPhone. Set `HOST=0.0.0.0` in `.env` and restart.
3. **Same Wi-Fi network?** The iPhone must be on the same network as the Mac,
   and not on cellular. Turn off **Private Relay** and any VPN on the iPhone;
   both silently break `.local` addressing.
4. **Mac asleep?** A sleeping MacBook does not answer. Keep the lid open, or
   **System Settings → Battery → Options → Prevent automatic sleeping when the
   display is off** (on power).
5. **Firewall.** **System Settings → Network → Firewall**. Either turn it off on
   your home network, or leave it on and add an allow rule — see
   [`local-network.md`](local-network.md#macos-firewall).
6. **`.local` name wrong?** On the Mac: `scutil --get LocalHostName`. Use exactly
   that, plus `.local`. If Bonjour is flaky on your network, fall back to the IP
   address (`ipconfig getifaddr en0`) and set a DHCP reservation on your router.

### It speaks JSON, or says something like "answer"

The **Get Dictionary Value** step isn't wired up. Check it says `Value` for
`answer` **in** `Contents of URL`, and that **Speak Text** references the
*Dictionary Value*, not *Contents of URL*.

### It says "Missing or invalid bearer token"

The `Authorization` header is wrong. It must be exactly `Bearer ` + the token
from `.env` — one space, no quotes, no trailing newline. Retype it rather than
pasting, since iOS loves to add a trailing space. Confirm with `curl` from the
Mac first:

```
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Authorization: Bearer $LOCAL_ASSISTANT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"hello","source":"api"}'
```

### It speaks "The local model isn't available" (or `/health` says degraded)

The Ollama runtime is down or the model isn't pulled.

```
ollama list                 # is qwen3:8b there?
ollama pull qwen3:8b        # if not
ollama serve                # if the runtime isn't running at all
curl http://127.0.0.1:11434/api/tags
```

`GET /health` on the Mac tells you which of the two it is: `runtime_ok: false`
means Ollama is down, `model_ok: false` means the model isn't pulled.

### First question of the day is very slow, later ones are fast

Expected. Ollama loads ~5 GB of weights from disk on a cold start. `LLM_KEEP_ALIVE`
(default `10m`) controls how long it stays resident. Raise it to `30m` if you use
the assistant in bursts; lower it if you want the RAM back sooner.

### It says it couldn't find current information

The web search or page fetches failed or came back too thin. Check the Mac's
console log for the `ask route=web` line — it reports how many results and pages
were used. Common causes: no internet, DuckDuckGo rate-limiting you (switch
`SEARCH_PROVIDER` to `brave` with a free API key), or the pages were all
JavaScript-only apps with no extractable text.

### The answer is right but the voice sounds Mandarin

The **Speak Text** voice is set to a Mandarin voice. Change it to Sinji or
another Cantonese (Hong Kong) voice. This is purely a Shortcut setting — the
backend only ever returns text.

### The answer is too long to listen to

Lower `LLM_MAX_TOKENS_SIRI` in `.env` (default 400). The prompt already asks for
2–5 sentences, but a smaller ceiling enforces it.
