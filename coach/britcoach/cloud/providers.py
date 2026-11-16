"""Optional cloud LLM providers (Anthropic, OpenAI) for advanced analysis only.

SDKs are imported lazily and keys come only from the environment
(``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``). Nothing here is used for ordinary
conversation turns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class CloudError(RuntimeError):
    pass


class CloudProvider(ABC):
    name = "base"
    model = ""

    @abstractmethod
    async def complete(self, system: str, user: str, max_tokens: int = 2000) -> str: ...


class AnthropicProvider(CloudProvider):
    name = "anthropic"

    def __init__(self, model: str, timeout_s: float = 60.0, client=None) -> None:
        self.model = model
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic(timeout=timeout_s)  # reads ANTHROPIC_API_KEY
        self._client = client

    async def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        import anthropic

        try:
            resp = await self._client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low"},  # short, well-specified analysis task
                # If the model declines, the API re-runs the request on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise CloudError("Anthropic rejected the API key (check ANTHROPIC_API_KEY).") from e
        except anthropic.RateLimitError as e:
            raise CloudError("Anthropic rate limit reached; try again shortly.") from e
        except anthropic.APIStatusError as e:
            raise CloudError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise CloudError("Could not reach the Anthropic API.") from e
        if resp.stop_reason == "refusal":
            raise CloudError("The model declined this request.")
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAIProvider(CloudProvider):
    name = "openai"

    def __init__(self, model: str, timeout_s: float = 60.0, client=None) -> None:
        self.model = model
        if client is None:
            import openai

            client = openai.AsyncOpenAI(timeout=timeout_s)  # reads OPENAI_API_KEY
        self._client = client

    async def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        import openai

        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=max_tokens,
            )
        except openai.AuthenticationError as e:
            raise CloudError("OpenAI rejected the API key (check OPENAI_API_KEY).") from e
        except openai.RateLimitError as e:
            raise CloudError("OpenAI rate limit reached; try again shortly.") from e
        except openai.APIStatusError as e:
            raise CloudError(f"OpenAI API error {e.status_code}") from e
        except openai.APIConnectionError as e:
            raise CloudError("Could not reach the OpenAI API.") from e
        return resp.choices[0].message.content or ""
