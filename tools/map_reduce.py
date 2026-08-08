"""
Budget-aware map-reduce compression.
 
Used in two places by tools/retrieval.py:
  1. Pre-RAG: compress a raw filing before chunking/embedding, when the raw
     document is too large to index efficiently or is dominated by noise.
  2. Post-retrieval: compress retrieved chunks further if, combined, they
     still exceed the model's real context budget.
 
Design principles:
  - Never hardcode a token limit -- query the model itself (get_model_context_limit).
  - Never trust a char-count heuristic for a hard budget check -- use a real
    tokenizer (count_tokens).
  - Never loop indefinitely -- a no-progress guard in compress_to_fit
    guarantees termination even if compression stalls or plateaus.
"""

import logging

import ollama
import tiktoken

from tools.edgar import chunk_text

CHAT_MODEL = "qwen2.5:14b"

_encoder = tiktoken.get_encoding("cl100k_base1")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token counting and context-limit detection
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Real tokenizer count. Not qwen's exact tokenizer, but BPE tokenizers
    are close enough to use as a budget check with a safety margin."""
    return len(_encoder.encode(text))


def get_model_context_limit(model: str = CHAT_MODEL, default: int = 8192) -> int:
    """Query Ollama for the model's actual context window rather than guessing.
    Falls back to `default` if the lookup fails -- fail safe, not crash."""
    try:
        info = ollama.show(model)
        model_info = info.get("model_info", {})
        # key is family-specific, e.g. "qwen2.context_length"
        for key, value in model_info.items():
            if key.endswith("context_length"):
                return int(value)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not determine context limit for {model}: {e}")
    return default # fail safe rather than crash


# ---------------------------------------------------------------------------
# Map step -- one LLM call per chunk, extracting only what's relevant to the
# goal. This is what makes it "map-reduce" rather than naive truncation:
# every chunk gets a chance to contribute, nothing is silently dropped, and
# irrelevant boilerplate gets filtered before it competes for space.
# ---------------------------------------------------------------------------

def _map_chunk(chunk: str, goal: str, model: str = CHAT_MODEL) -> str:
    prompt = f"""Original research goal: {goal}

Extract ONLY the information below relevant to that goal. Be concise.
If nothing here is relevant, respond with exactly: NOT_RELEVANT

Text:
{chunk}"""
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Reduce loop -- iteratively map+reduce text until it fits the model's real
# context budget, with a no-progress guard so a stalled or worsening
# compression pass can't loop forever.
# ---------------------------------------------------------------------------

def compress_to_fit(
    text_or_chunks: str | list[str],
    goal: str,
    model: str = CHAT_MODEL,
    response_reserve: int = 1500,
    safety_margin: float = 0.85,
    max_passes: int = 4,
) -> str:
    """
    Iteratively compress `text_or_chunks` (raw text, or a list of chunks to
    join) down to fit under the model's real context budget, guided by goal.
 
    response_reserve: tokens to leave headroom for the model's own reply.
    safety_margin: fraction of the raw context window to actually use
                   (leaves room for tokenizer mismatch between tiktoken's
                   approximation and the model's real tokenizer).
    max_passes: hard ceiling on map-reduce iterations, regardless of progress.
    """
    limit = get_model_context_limit(model)
    budget = int(limit * safety_margin) - response_reserve

    text = "\n\n".join(text_or_chunks) if isinstance(text_or_chunks, list) else text_or_chunks
    prev_tokens = count_tokens(text)

    for pass_num in range(max_passes):
        if prev_tokens <= budget:
            return text

        pieces = chunk_text(text, chunk_size=3000)
        extracts = [
            r for p in pieces
            if (r := _map_chunk(p, goal, model)) != "NOT_RELEVANT"
        ]
        new_text = "\n\n".join(extracts) if extracts else text
        new_tokens = count_tokens(new_text)

        # No-progress guard: if this pass barely helped (or made things
        # worse), stop rather than keep burning LLM calls for no benefit.
        if new_tokens >= prev_tokens * 0.95:
            logger.warning(
                f"map-reduce compression stalled at pass {pass_num} "
                f"({prev_tokens}->{new_tokens} tokens); returning best effort"
            )
            return new_text

        text, prev_tokens = new_text, new_tokens

    logger.warning(
        f"map-reduce hit max_passes={max_passes} still over budget "
        f"({prev_tokens} > {budget} tokens)"
    )
    return text