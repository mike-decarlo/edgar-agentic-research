"""CLI entry point for the SEC filing research agent.

Prompts for a ticker, runs the researcher -> verifier pipeline, prints the final
answer, and writes a full audit-trail JSON for the run.
"""

import logging

from agents.verifier import run_pipeline_with_retry
from audit import write_audit_log
from tools.edgar import TickerNotFoundError


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_goal(ticker: str) -> str:
    return (
        f"Look up {ticker}'s most recent 10-K filing and its NetIncomeLoss "
        "XBRL history. Summarize the trend in 3 sentences and flag anything odd "
        "about data quality or gaps in the reported values."
    )


def main() -> None:
    configure_logging()
    log = logging.getLogger("main")

    ticker = input("Enter a stock ticker (e.g. AAPL, GOOGL): ").strip().upper()
    if not ticker:
        print("No ticker entered. Exiting.")
        return

    goal = build_goal(ticker)

    try:
        result = run_pipeline_with_retry(goal)
    except TickerNotFoundError as exc:
        # Private companies, funds/ETFs, and non-US listings won't resolve.
        print(f"\nCould not research {ticker}: {exc}")
        return
    except Exception as exc:  # network/API failures — fail readably, not with a traceback
        log.exception("pipeline failed")
        print(f"\nSomething went wrong while researching {ticker}: {exc}")
        return

    audit_path = write_audit_log(ticker, goal, result)

    print("\n=== FINAL ANSWER ===")
    print(result.final_answer)
    status = "passed verification" if result.passed else "did NOT pass verification"
    print(f"\n({status} after {result.attempts} attempt(s). Audit log: {audit_path})")


if __name__ == "__main__":
    main()
