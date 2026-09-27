"""LLM client for the Digital Twin.

Primary provider: OpenRouter (nemotron model, see TWIN_MODEL).
Fallback provider: Groq (only active when GROQ_API_KEY is set in .env).

When the primary provider is rate-limited, down, or returns unusable
content, the same conversation is automatically retried against the
fallback. Never logs or exposes any API key.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from .config import (
    API_KEY,
    DEPLOY_MODE,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_URL,
    MODEL,
    OPENROUTER_URL,
    REFERER,
)

_TIMEOUT = httpx.Timeout(90.0)


class TwinLLMError(Exception):
    """Raised when model calls fail; message is safe to show users."""


# ---------------------------------------------------------------- plumbing


def _providers() -> list[dict]:
    """Ordered provider configs; only those with configured keys."""
    providers: list[dict] = []
    if API_KEY:
        providers.append(
            {"name": "openrouter", "api_key": API_KEY, "model": MODEL, "url": OPENROUTER_URL}
        )
    if GROQ_API_KEY:
        providers.append(
            {"name": "groq", "api_key": GROQ_API_KEY, "model": GROQ_MODEL, "url": GROQ_URL}
        )
    return providers


def _headers_for(provider: dict) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    if provider["name"] == "openrouter" and REFERER:
        headers["HTTP-Referer"] = REFERER
        headers["X-Title"] = "Padmanav Mohanty - Digital Twin"
    return headers


def _payload_for(provider: dict, messages: list[dict[str, str]], stream: bool) -> dict:
    return {
        "model": provider["model"],
        "messages": messages,
        "max_tokens": 1600,
        "temperature": 0.4,
        **({"stream": True} if stream else {}),
    }


def _friendly_error(status: int) -> str:
    if status == 401:
        return "Authentication with a model provider failed."
    if status == 402:
        return "A model provider reported insufficient credits."
    if status == 429:
        return "Rate limit reached — please wait a moment and try again."
    if status in (408, 504):
        return "The model took too long to respond. Please try again."
    if status == 503:
        return "The model provider is temporarily unavailable. Please try again soon."
    if status >= 500:
        return "The model provider had a problem. Please try again shortly."
    return f"The model provider rejected the request (error {status})."


_NOT_CONFIGURED = (
    "The Digital Twin is not configured yet: the Space owner needs to set "
    "OPENROUTER_API_KEY (and optionally GROQ_API_KEY) in the Space secrets."
    if DEPLOY_MODE
    else (
        "The Digital Twin is not configured yet: add OPENROUTER_API_KEY (and "
        "optionally GROQ_API_KEY for fallback) to the .env file and restart."
    )
)


# ------------------------------------------------------------- single-shot


async def _complete_once(provider: dict, messages: list[dict[str, str]]) -> str:
    payload = _payload_for(provider, messages, stream=False)

    for attempt in (1, 2):  # one internal retry per provider
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    provider["url"], headers=_headers_for(provider), json=payload
                )
        except httpx.TimeoutException as exc:
            raise TwinLLMError("The model took too long to respond. Please try again.") from exc
        except httpx.HTTPError as exc:
            raise TwinLLMError("Could not reach the model provider. Please try again.") from exc

        if resp.status_code == 429 and attempt == 1:
            await asyncio.sleep(6)  # transient congestion is usually brief
            payload["max_tokens"] = 3200
            continue
        if resp.status_code >= 400:
            raise TwinLLMError(_friendly_error(resp.status_code))

        try:
            data = resp.json()
            message = data["choices"][0].get("message", {})
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise TwinLLMError("The model returned an unexpected response.") from exc

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        # Reasoning models can exhaust the budget on hidden reasoning and
        # return missing/null/empty content; retry once with a larger budget.
        if attempt == 1:
            payload["max_tokens"] = 3200
            continue

    raise TwinLLMError("The model returned an empty response. Please try again.")


async def generate_reply(messages: list[dict[str, str]]) -> str:
    """Send the conversation to the primary provider; fall back if it fails."""
    providers = _providers()
    if not providers:
        raise TwinLLMError(_NOT_CONFIGURED)

    last_error: TwinLLMError | None = None
    for provider in providers:
        try:
            return await _complete_once(provider, messages)
        except TwinLLMError as exc:
            last_error = exc  # try the next provider
    raise last_error or TwinLLMError("All model providers failed. Please try again.")


# ---------------------------------------------------------------- streaming


async def _stream_once(
    provider: dict, messages: list[dict[str, str]]
) -> AsyncIterator[str]:
    payload = _payload_for(provider, messages, stream=True)

    for attempt in (1, 2):  # one internal retry per provider
        produced = 0
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                async with client.stream(
                    "POST", provider["url"], headers=_headers_for(provider), json=payload
                ) as resp:
                    if resp.status_code == 429 and attempt == 1:
                        await resp.aread()
                        await asyncio.sleep(6)
                        payload["max_tokens"] = 3200
                        continue
                    if resp.status_code >= 400:
                        await resp.aread()
                        raise TwinLLMError(_friendly_error(resp.status_code))

                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                        except ValueError:
                            continue
                        delta = (
                            event.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content")
                        )
                        if isinstance(delta, str) and delta:
                            produced += 1
                            yield delta
        except httpx.TimeoutException as exc:
            raise TwinLLMError("The model took too long to respond. Please try again.") from exc
        except httpx.HTTPError as exc:
            raise TwinLLMError("Could not reach the model provider. Please try again.") from exc

        if produced:
            return
        # No visible content (reasoning exhausted the budget): retry larger.
        if attempt == 1:
            payload["max_tokens"] = 3200
            continue

    raise TwinLLMError("The model returned an empty response. Please try again.")


async def generate_reply_stream(
    messages: list[dict[str, str]],
) -> AsyncIterator[str]:
    """Stream the reply token-by-token, falling back across providers.

    A provider is only swapped while nothing has been yielded yet, so the
    user never sees duplicated or restarted text mid-answer.
    """
    providers = _providers()
    if not providers:
        raise TwinLLMError(_NOT_CONFIGURED)

    last_error: TwinLLMError | None = None
    for provider in providers:
        emitted = 0
        try:
            async for delta in _stream_once(provider, messages):
                emitted += 1
                yield delta
            return
        except TwinLLMError as exc:
            last_error = exc
            if emitted:
                raise  # partial reply already shown; restarting would duplicate it
            continue  # nothing shown yet — silently try the fallback provider

    raise last_error or TwinLLMError("All model providers failed. Please try again.")
