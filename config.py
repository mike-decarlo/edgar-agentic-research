"""Shared configuration: the Ollama model and the SEC EDGAR request headers.

The SEC EDGAR API requires every request to carry a User-Agent that identifies
who is making it (name + contact email). We read that from the environment
(``SEC_USER_AGENT``) rather than hardcoding it, so no real email ever lands in
a committed file. Copy ``.env.example`` to ``.env`` and fill it in.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Loading .env is a convenience, not a hard requirement. If python-dotenv isn't
# installed, we simply rely on the process environment being set another way.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - trivial fallback
    logger.debug("python-dotenv not installed; reading environment directly.")

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

# Hosted fallback model (Groq's OpenAI-compatible API) for GPU-free deployments.
# See llm.py for how the provider is chosen. Any tool-capable open-weight model
# on Groq works; llama-3.3-70b-versatile is a reliable default.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# The placeholder shipped in .env.example — treated as "not configured".
_PLACEHOLDER_UA = "Your Name your-email@example.com"


def get_user_agent() -> str:
    """Return the configured SEC User-Agent, or raise a helpful error."""
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua or ua == _PLACEHOLDER_UA:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. The SEC EDGAR API rejects requests "
            "without a User-Agent identifying you (e.g. 'Jane Doe "
            "jane@example.com'). Copy .env.example to .env and set "
            "SEC_USER_AGENT, or export it in your shell."
        )
    return ua


def get_headers() -> dict[str, str]:
    """Request headers for every SEC EDGAR call."""
    return {"User-Agent": get_user_agent()}
