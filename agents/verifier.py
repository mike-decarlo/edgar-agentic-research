"""Agent 2 — Verifier — plus the orchestration that ties both agents together.

Two layers of checking, cheapest first:

1. ``check_numeric_sanity`` — deterministic arithmetic, no LLM. Catches
   magnitude errors (wrong digit, bad dedupe, unit mismatch) far more reliably
   than asking a model to eyeball raw numbers in JSON. Code owns arithmetic.

2. ``verify_answer`` — a second model call with NO tools, whose narrow job is to
   compare the researcher's prose against the raw data it collected and flag
   unsupported claims. The LLM owns semantic/reasoning checks, not arithmetic.

The orchestrator runs (1) as a gate before (2): if the numbers are already
impossible, there is no point spending an LLM call to check the wording.
"""

import json
import logging
from dataclasses import dataclass, field

from ollama import chat

from agents.researcher import english_system_prompt, run_researcher
from config import MODEL

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything a single pipeline run produced — the audit trail's payload."""

    final_answer: str
    tool_log: list[dict]
    numeric_issues: list[str] = field(default_factory=list)
    verifier_check: dict | None = None
    attempts: int = 1
    passed: bool = False


def check_numeric_sanity(tool_log: list[dict]) -> list[str]:
    """Deterministic magnitude check over the researcher's XBRL results.

    Flags any year-over-year move smaller than ~1/3 or larger than ~3x, which
    for a stable balance-sheet concept almost always means a data or scale
    error (e.g. a duplicated fiscal year that wasn't deduped) rather than a
    real change.
    """
    issues = []
    for entry in tool_log:
        if entry["tool"] != "get_xbrl_fact":
            continue
        values = entry["result"].get("values", [])
        for prev, curr in zip(values, values[1:]):
            v1, v2 = prev.get("val"), curr.get("val")
            if v1 and v2 and (v2 < v1 * 0.3 or v2 > v1 * 3):
                issues.append(
                    f"{entry['result']['concept']}: {prev['end']}={v1:,} -> "
                    f"{curr['end']}={v2:,} is a >3x swing -- likely a data or "
                    "scale error"
                )
    return issues


def verify_answer(user_goal: str, agent_answer: str, tool_log: list[dict]) -> dict:
    """LLM fact-check of the researcher's prose against its own raw data."""
    check_prompt = f"""You are a strict fact-checker. Do not do new research.

    Original request: {user_goal}

    Agent's answer:
    {agent_answer}

    Raw data the agent actually collected (ground truth):
    {json.dumps(tool_log, default=str)[:3000]}

    Check the agent's answer against ONLY the raw data above. Flag:
    - Any number or claim not directly supported by the raw data
    - Any non-English text or garbled output
    - Missing caveats about gaps/oddities that ARE visible in the raw data

    Respond with ONLY valid JSON, no other text:
    {{"valid": true/false, "issues": ["issue1", "issue2"], "corrected_answer": "..."}}
    """
    response = chat(
        model=MODEL,
        messages=[english_system_prompt(), {"role": "user", "content": check_prompt}],
    )
    raw = response["message"]["content"].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Model didn't return clean JSON — surface that as a verifier failure
        # itself, rather than silently trusting the unverified answer.
        logger.warning("verifier returned non-JSON output")
        return {
            "valid": False,
            "issues": [f"Verifier returned non-JSON output: {raw[:200]}"],
            "corrected_answer": agent_answer,
        }


def run_pipeline_with_retry(user_goal: str, max_retries: int = 2) -> PipelineResult:
    """Loop researcher -> checks, feeding issues back as corrections.

    The deterministic sanity gate runs first each attempt; if it fires we stop
    and return immediately, because a magnitude error is a data problem the
    LLM verifier cannot talk its way out of. Otherwise the LLM verifier runs,
    and any issues it finds are appended to the goal for the next attempt.
    """
    goal = user_goal
    last_check: dict | None = None

    for attempt in range(1, max_retries + 1):
        answer, tool_log = run_researcher(goal)
        logger.info("--- attempt %d: researcher answered ---\n%s", attempt, answer)

        numeric_issues = check_numeric_sanity(tool_log)
        if numeric_issues:
            logger.warning(
                "deterministic sanity check flagged issues (skipping LLM "
                "verifier): %s",
                numeric_issues,
            )
            return PipelineResult(
                final_answer=(
                    "Flagged before verification by the deterministic sanity "
                    f"check: {numeric_issues}"
                ),
                tool_log=tool_log,
                numeric_issues=numeric_issues,
                attempts=attempt,
                passed=False,
            )

        check = verify_answer(goal, answer, tool_log)
        last_check = check
        logger.info("--- verifier check ---\n%s", json.dumps(check, indent=2))

        if check.get("valid"):
            return PipelineResult(
                final_answer=answer,
                tool_log=tool_log,
                verifier_check=check,
                attempts=attempt,
                passed=True,
            )

        # Feed the verifier's issues back into the researcher and try again.
        goal = (
            f"{user_goal}\n\nA reviewer found these problems with a previous "
            f"attempt -- fix them: {check.get('issues')}"
        )

    logger.warning("pipeline exhausted %d attempts without passing", max_retries)
    return PipelineResult(
        final_answer=(last_check or {}).get("corrected_answer", answer),
        tool_log=tool_log,
        verifier_check=last_check,
        attempts=max_retries,
        passed=False,
    )
