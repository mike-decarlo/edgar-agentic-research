"""Agent 1 — Researcher.

A ReAct-style tool loop: call the model, execute any tool calls it requests,
feed the results back, and repeat until it answers in plain text. Alongside the
final answer it returns a ``tool_log`` — every (tool, args, result) triple — so
the verifier and the deterministic sanity check have the researcher's own raw
data to check against.
"""

import json
import logging
from datetime import date

from ollama import chat

from config import MODEL
from tools.edgar import TOOL_FUNCTIONS, TOOL_SCHEMAS

logger = logging.getLogger(__name__)


def english_system_prompt() -> dict:
    """System message shared by both agents.

    Includes today's date so the model doesn't mistake recent SEC filing dates
    (e.g. a 2025 10-K) for impossible "future" dates — a knowledge-cutoff
    artifact that otherwise makes the verifier reject valid data.
    """
    return {
        "role": "system",
        "content": (
            f"Today's date is {date.today().isoformat()}. Dates on or before "
            "today are in the past and are valid. You must always respond in "
            "English only."
        ),
    }


def run_researcher(user_goal: str, max_turns: int = 6) -> tuple[str, list[dict]]:
    """Run the researcher loop; return (final_answer, tool_log)."""
    messages = [english_system_prompt(), {"role": "user", "content": user_goal}]
    tool_log: list[dict] = []

    for turn in range(max_turns):
        response = chat(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        msg = response["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return msg.get("content", ""), tool_log

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]  # already a dict in ollama's client
            logger.info("[turn %d] calling %s(%s)", turn, name, args)
            try:
                result = TOOL_FUNCTIONS[name](**args)
            except Exception as exc:  # surfaced back to the model as a tool result
                logger.warning("tool %s failed: %s", name, exc)
                result = {"error": str(exc)}

            tool_log.append({"tool": name, "args": args, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(result, default=str)[:4000],  # cap size
                }
            )

    logger.warning("researcher hit max_turns (%d) without a final answer", max_turns)
    return "Max turns reached without a final answer.", tool_log
