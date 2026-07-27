"""Persist each pipeline run to a timestamped JSON file as an audit trail.

Every run writes the goal, the final answer, the verifier's verdict, and the
full raw tool log to ``logs/<UTC timestamp>_<TICKER>.json`` so any answer can
later be traced back to the exact SEC data it was derived from.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from agents.verifier import PipelineResult

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent / "logs"


def write_audit_log(
    ticker: str, goal: str, result: PipelineResult, log_dir: Path | str = LOG_DIR
) -> Path:
    """Write one run's full record to a timestamped JSON file; return its path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = log_dir / f"{ts}_{ticker}.json"

    payload = {
        "timestamp": ts,
        "ticker": ticker,
        "goal": goal,
        "passed": result.passed,
        "attempts": result.attempts,
        "final_answer": result.final_answer,
        "numeric_issues": result.numeric_issues,
        "verifier_check": result.verifier_check,
        "tool_log": result.tool_log,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote audit log to %s", path)
    return path
