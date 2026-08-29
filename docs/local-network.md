# Reaching the Mac from the iPhone

The backend runs on the MacBook. The iPhone has to find it and be allowed to
talk to it. Three things: listen on the LAN, have a stable address, get past the
firewall.

## 1. Listen on the LAN, not just localhost

`HOST=0.0.0.0` in `.env` (the default). `127.0.0.1` means "this Mac only" and
the iPhone will simply time out.

Verify:

```bash
lsof -nP -iTCP:8000 | grep LISTEN
# want:  Python ... TCP *:8000 (LISTEN)
# not:   Python ... TCP 127.0.0.1:8000 (LISTEN)
```

## 2. A stable address

Three options, simplest first.

### Bonjour `.local` hostname — recommended

macOS already advertises itself. Find the name:

```bash
scutil --get LocalHostName        # e.g. "Charles-MacBook-Air"
```

Your URL is then `http://Charles-MacBook-Air.local:8000`. Test it from the
iPhone's Safari before touching Shortcuts.

This survives DHCP lease changes and router reboots, needs no configuration, and
is the right default. Rename the Mac to something short and typo-proof under
**System Settings → General → Sharing → Local hostname** if you like.

Bonjour breaks on some networks — guest Wi-Fi, networks with AP/client isolation,
and some mesh systems that don't forward mDNS between bands. If `.local` doesn't
resolve, use option 2.

### DHCP reservation — the fallback

Find the current IP:

```bash
ipconfig getifaddr en0            # Wi-Fi
```

Then in your router's admin page, reserve that IP for the Mac's MAC address. The
address stops changing and you can hard-code `http://192.168.1.42:8000` in the
Shortcut. Works everywhere Bonjour doesn't; costs you a router config, and breaks
if you move to a different network.

### Tailscale — only if you want it away from home

Install Tailscale on both devices; the Mac gets a permanent `100.x.y.z` address
and a MagicDNS name that works from anywhere, over an encrypted WireGuard tunnel.

This is the right answer *if* you want to ask your assistant questions while out
of the house. For Phase 1 — same room, same Wi-Fi — it is a background daemon,
a battery cost on the phone, and an account, in exchange for nothing. Start with
Bonjour. Add Tailscale the day you actually want remote access, and when you do,
it is strictly better than port-forwarding your router (which you should never
do with this service).

## 3. macOS firewall

**System Settings → Network → Firewall.**

If it's off, nothing to do. If it's on, macOS prompts *"Do you want the
application 'Python' to accept incoming network connections?"* the first time you
start the server — click **Allow**. If you missed the prompt or it never
appeared:

**Firewall → Options…** → **+** → navigate to the Python binary inside the
project's venv (`<project>/.venv/bin/python3` — use ⌘⇧G in the file picker to
type the path) → set it to **Allow incoming connections**.

Two settings that break things quietly, both under **Options…**:

- **Block all incoming connections** — blocks this too. Turn it off.
- **Enable stealth mode** — fine for TCP, but it drops the pings you'd use to
  debug. Not a blocker, just don't be confused by failing pings.

## 4. Test from the iPhone

In Safari on the iPhone:

```
http://your-mac.local:8000/health
```

Expected:

```json
{"status":"ok","backend":true,"llm":{"provider":"ollama","model":"qwen3:8b",
"runtime_ok":true,"model_ok":true,"detail":""},"search_provider":"duckduckgo"}
```

`/health` needs no token, which is exactly why it is the right thing to test
with. `status: degraded` means the HTTP path is fine and the *model* is the
problem — read `llm.runtime_ok` and `llm.model_ok`.

If Safari can't load it, work through the checks in
[`apple-shortcut.md`](apple-shortcut.md#could-not-connect-to-the-server--the-shortcut-hangs-then-fails).

## 5. A note on exposure

This service answers to anyone on your Wi-Fi who has the bearer token, and it is
plain HTTP — the token crosses your LAN in the clear. On a home network that is a
reasonable trade for zero setup friction. What you should not do:

- **Do not port-forward it** to the public internet.
- **Do not run it on untrusted Wi-Fi** (cafés, hotels, conferences) without
  Tailscale, since anyone on the same network can see the token.
- **Do not commit `.env`.** It is gitignored; keep it that way.

Rotate the token by editing `.env`, restarting, and updating the Shortcut header.
