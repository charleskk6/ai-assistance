"""Cloud routing policy.

Cloud processing is **off by default** and is only ever used for analysis, never
for the conversation itself. A call is allowed only when all of these hold:

* the user has enabled cloud processing (globally via ``COACH_CLOUD_ENABLED`` or
  per session via the UI toggle), and an API key for the provider is present;
* the purpose is permitted:
    - ``review``: the user explicitly asked for an advanced (cloud) review *and*
      gave consent in that request, or the escalation policy is
      ``local_failure`` and the local analysis failed (still requires consent);
    - ``turn_correction``: the user enabled cloud-assisted corrections;
    - ``conversation``: never.

Every call is announced to the UI (start/end events) and logged to SQLite with
the provider, purpose and number of characters sent.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..config import Settings
from .providers import AnthropicProvider, CloudError, CloudProvider, OpenAIProvider

Purpose = str  # "conversation" | "review" | "turn_correction"


@dataclass
class CloudPrefs:
    """Per-session cloud preferences (from the UI)."""

    enabled: bool = False
    provider: str = "anthropic"
    allow_turn_corrections: bool = False


@dataclass
class RouteDecision:
    allowed: bool
    reason: str
    provider: str = ""


class CloudRouter:
    def __init__(self, settings: Settings, provider_factory: Callable[[str], CloudProvider] | None = None,
                 on_event: Callable[[dict], Awaitable[None]] | None = None,
                 audit: Callable[[dict], Awaitable[None]] | None = None) -> None:
        self.settings = settings
        self._factory = provider_factory or self._default_factory
        self._providers: dict[str, CloudProvider] = {}
        self.on_event = on_event
        self.audit = audit

    def _default_factory(self, name: str) -> CloudProvider:
        s = self.settings
        if name == "anthropic":
            return AnthropicProvider(s.anthropic_model, s.cloud_timeout_s)
        if name == "openai":
            return OpenAIProvider(s.openai_model, s.cloud_timeout_s)
        raise CloudError(f"Unknown cloud provider {name}")

    def key_present(self, provider: str) -> bool:
        return bool(self.settings.anthropic_api_key if provider == "anthropic" else self.settings.openai_api_key)

    def decide(self, purpose: Purpose, prefs: CloudPrefs, explicit_request: bool = False,
               consent: bool = False, local_failed: bool = False) -> RouteDecision:
        s = self.settings
        provider = prefs.provider or s.cloud_provider
        if purpose == "conversation":
            return RouteDecision(False, "ordinary conversation always stays local")
        if not (s.cloud_enabled or prefs.enabled):
            return RouteDecision(False, "cloud processing is disabled")
        if not self.key_present(provider):
            return RouteDecision(False, f"no API key for {provider} in the environment")
        if purpose == "review":
            if not s.cloud_allow_review:
                return RouteDecision(False, "cloud review is disabled by configuration")
            if not consent:
                return RouteDecision(False, "the transcript is only uploaded with explicit consent")
            if explicit_request:
                return RouteDecision(True, "user requested advanced analysis", provider)
            if local_failed and s.cloud_escalation == "local_failure":
                return RouteDecision(True, "escalated: local analysis failed", provider)
            return RouteDecision(False, "escalation policy does not permit this call")
        if purpose == "turn_correction":
            if prefs.allow_turn_corrections or s.cloud_allow_turn_corrections:
                return RouteDecision(True, "cloud-assisted corrections enabled", provider)
            return RouteDecision(False, "cloud-assisted corrections are not enabled")
        return RouteDecision(False, f"unknown purpose {purpose}")

    async def complete(self, decision: RouteDecision, purpose: Purpose, system: str, user: str,
                       max_tokens: int = 2000) -> str:
        if not decision.allowed:
            raise CloudError(f"Cloud call not permitted: {decision.reason}")
        provider = self._providers.get(decision.provider) or self._factory(decision.provider)
        self._providers[decision.provider] = provider
        info = {"provider": provider.name, "model": provider.model, "purpose": purpose,
                "chars_sent": len(system) + len(user), "reason": decision.reason}
        if self.on_event:
            await self.on_event({"type": "cloud_status", "active": True, **info})
        t0 = time.perf_counter()
        ok = False
        try:
            out = await provider.complete(system, user, max_tokens)
            ok = True
            return out
        finally:
            info["latency_s"] = round(time.perf_counter() - t0, 3)
            info["ok"] = ok
            if self.on_event:
                await self.on_event({"type": "cloud_status", "active": False, **info})
            if self.audit:
                await self.audit(info)
