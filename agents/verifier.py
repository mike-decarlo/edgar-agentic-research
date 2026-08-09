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
from datetime import date

from ollama import chat

from agents.researcher import english_system_prompt, run_researcher
from config import MODEL
from tools.edgar import get_xbrl_fact

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


def check_numeric_sanity(tool_log: list[dict], client_fxn=get_xbrl_fact) -> list[str]:
    """
    Cross-validates NetIncomeLoss against Revenues using accounting identities
    (margin bounds), rather than flagging on raw magnitude swings alone.
    Magnitude swings are only flagged if they're ALSO not corroborated by
    revenue movement in the same direction -- catching data bugs (duplicate/
    misaligned fiscal periods) without penalizing genuine business growth.
    """
    issues = []
    net_income_entries = None
    revenue_entries = None

    for entry in tool_log:
        if entry["tool"] == "get_xbrl_fact" and entry["result"].get("concept") == "NetIncomeLoss":
            net_income_entries = entry["result"].get("values", [])
        if entry["tool"] == "get_xbrl_fact" and entry["result"].get("concept") == "Revenues":
            revenue_entries = entry["result"].get("values", [])

    if not net_income_entries:
        return issues

    # Check 1: fiscal period spacing -- catches the ORIGINAL bug class
    # (duplicate/misaligned fiscal years from undeduped SEC data), independent
    # of magnitude entirely.
    for prev, curr in zip(net_income_entries, net_income_entries[1:]):
        d1 = date.fromisoformat(prev["end"])
        d2 = date.fromisoformat(curr["end"])
        days = (d2 - d1).days
        if not (330 <= days <= 400): # should be roughly 1 fiscal year
            issues.append(
                f"NetIncomeLoss fiscal periods {prev['end']} -> {curr['end']} "
                f"are {days} days apart, not ~1 year -- possible duplicate/misaligned entry"
            )

    # Check 2: margin plausibility, if we have Revenues to cross-check against
    # Check 3: large swings are only suspicious if revenue moved the OPPOSITE
    # direction -- a real red flag (profit tripled while revenue fell?, as
    # opposed to profit growing faster than revenue (normal margin exapansion).
    if revenue_entries:
        rev_by_end = {v["end"]: v["val"] for v in revenue_entries}
        for ni in net_income_entries:
            rev = rev_by_end.get(ni["end"])
            if rev and rev > 0:
                margin = ni["val"] / rev
                # net income can't exceed revenue
                # allow generous loss room for bad years
                if margin > 1.0 or margin < -3.0:
                    issues.append(
                        f"NetIncomeLoss/Revenues margin for FY ending {ni['end']} "
                        f"is {margin:.1%} -- outside plausible bounds"
                    )

        for prev, curr in zip(net_income_entries, net_income_entries[1:]):
            rev_prev = rev_by_end.get(prev["end"])
            rev_curr = rev_by_end.get(curr["end"])
            if rev_prev and rev_curr:
                ni_grew = curr["val"] > prev["val"] * 1.5
                rev_shrank = rev_curr < rev_prev * 0.95
                if ni_grew and rev_shrank:
                    issues.append(
                        f"NetIncomeLoss grew sharply ({prev['end']}->{curr['end']}) "
                        f"while Revenues declined -- inconsistent, worth reviewing"
                    )

    else:
        issues.append(
            "NetIncomeLoss reported without Revenues to cross-check margin "
            "plausibility -- researcher did not pull Revenue; handle with caution"
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
