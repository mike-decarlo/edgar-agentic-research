"""Provider-agnostic chat-completion wrapper.

Local dev runs Ollama against a GPU (default ``qwen2.5:14b``). Free public
hosting (e.g. Streamlit Community Cloud) has no GPU, so the same completion
calls fall back to Groq's OpenAI-compatible API with an equivalent open-weight
model.

Only the *completion call* is abstracted here. The researcher's ReAct loop, the
``tool_log``, the deterministic sanity gate, and the LLM verifier are all
unchanged — they call :func:`chat` exactly the way they used to call
``ollama.chat`` and get back the same ``{"message": {...}}`` shape.

Normalized message shape returned by :func:`chat`::

    {"message": {
        "role": "assistant",
        "content": "...",
        "tool_calls": [                       # absent when the model just answers
            {"id": "...", "function": {"name": "...", "arguments": {..dict..}}}
        ],
    }}

``arguments`` is always a ``dict`` (Ollama's native shape), so the ReAct loop
can splat it straight into a tool function regardless of provider.
"""

import json
import logging
import os
from collections import deque

from config import MODEL, GROQ_MODEL

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Resolved once per process; the choice can't change mid-run.
_provider: str | None = None
_groq_client = None


def _ollama_available() -> bool:
    """True if a local Ollama server answers. Cheap ping, failures are False."""
    try:
        import ollama

        ollama.list()
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "not available"
        logger.debug("Ollama not available: %s", exc)
        return False


def current_provider() -> str:
    """Resolve (and cache) which backend to use.

    ``LLM_PROVIDER`` forces a choice (``ollama``/``groq``); otherwise we auto-
    detect: prefer a running local Ollama, else Groq if an API key is present.
    """
    global _provider
    if _provider is not None:
        return _provider

    explicit = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    if explicit in ("ollama", "groq"):
        _provider = explicit
    elif _ollama_available():
        _provider = "ollama"
    elif os.getenv("GROQ_API_KEY"):
        _provider = "groq"
    else:
        raise RuntimeError(
            "No LLM backend available. Start a local Ollama server, or set "
            "GROQ_API_KEY (and optionally LLM_PROVIDER=groq) to use the hosted "
            "fallback."
        )

    logger.info("LLM provider resolved to %r", _provider)
    return _provider


def active_model() -> str:
    """The model name the resolved provider will actually use (for display)."""
    return GROQ_MODEL if current_provider() == "groq" else MODEL


def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Run one chat completion on the resolved provider.

    Returns ``{"message": <normalized-assistant-message>}``. ``tools`` uses the
    OpenAI-style ``{"type": "function", "function": {...}}`` envelope, which both
    Ollama and Groq accept unchanged.
    """
    if current_provider() == "ollama":
        return _ollama_chat(messages, tools)
    return _groq_chat(messages, tools)


# ---------------------------------------------------------------------------
# Ollama backend — returns the native message untouched, so the local path
# behaves exactly as it did before this wrapper existed.
# ---------------------------------------------------------------------------

def _ollama_chat(messages: list[dict], tools: list[dict] | None) -> dict:
    import ollama

    kwargs = {"model": MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    response = ollama.chat(**kwargs)
    return {"message": response["message"]}


# ---------------------------------------------------------------------------
# Groq backend (OpenAI-compatible). We translate the normalized conversation to
# OpenAI's schema on the way in and back on the way out, so the caller never
# sees the difference.
# ---------------------------------------------------------------------------

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from openai import OpenAI

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set; cannot use the Groq backend.")
        _groq_client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    return _groq_client


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert the normalized conversation to OpenAI/Groq wire format.

    OpenAI requires every ``tool`` result to carry the ``tool_call_id`` of the
    call it answers — something the provider-agnostic ReAct loop doesn't track.
    We reconstruct it: tool results are appended in the same order as the
    assistant's ``tool_calls``, so a FIFO of pending ids pairs them correctly.
    """
    out: list[dict] = []
    pending_ids: deque[str] = deque()

    for msg in messages:
        role = msg.get("role")
        tool_calls = msg.get("tool_calls") if role == "assistant" else None

        if tool_calls:
            converted = []
            for i, call in enumerate(tool_calls):
                cid = call.get("id") or f"call_{len(out)}_{i}"
                args = call["function"]["arguments"]
                converted.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": args if isinstance(args, str) else json.dumps(args),
                        },
                    }
                )
                pending_ids.append(cid)
            out.append(
                {"role": "assistant", "content": msg.get("content") or "", "tool_calls": converted}
            )
        elif role == "tool":
            cid = pending_ids.popleft() if pending_ids else "call_0"
            out.append({"role": "tool", "tool_call_id": cid, "content": msg.get("content", "")})
        else:
            out.append({"role": role, "content": msg.get("content", "")})

    return out


def _from_openai_message(message) -> dict:
    """Convert an OpenAI/Groq response message to the normalized shape."""
    normalized: dict = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        normalized["tool_calls"] = [
            {
                "id": tc.id,
                "function": {
                    "name": tc.function.name,
                    # Ollama hands the loop a dict; mirror that so the loop can
                    # splat arguments regardless of provider.
                    "arguments": json.loads(tc.function.arguments or "{}"),
                },
            }
            for tc in message.tool_calls
        ]
    return normalized


def _groq_chat(messages: list[dict], tools: list[dict] | None) -> dict:
    client = _get_groq_client()
    kwargs = {"model": GROQ_MODEL, "messages": _to_openai_messages(messages)}
    if tools:
        kwargs["tools"] = tools
    response = client.chat.completions.create(**kwargs)
    return {"message": _from_openai_message(response.choices[0].message)}
