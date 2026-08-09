"""CLI entry point for the SEC filing research agent.

Prompts for a ticker, runs the researcher -> verifier pipeline, prints the final
answer, and writes a full audit-trail JSON for the run.
"""

import logging

from agents.verifier import run_pipeline_with_retry
from audit import LOG_DIR, write_audit_log
from tools.edgar import TickerNotFoundError


def configure_logging() -> None:
    """Route all diagnostic logging to a file, keeping the console clean.

    The researcher, verifier, and httpx all emit useful INFO-level chatter, but
    it belongs in an audit file, not in front of the user. The user should see
    the agent work silently and then get a single final answer, so we attach a
    file handler to the root logger and deliberately add no console handler.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_DIR / "run.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def build_goal(ticker: str) -> str:
    return (
        f"Look up {ticker}'s most recent 10-K filing. Pull the historical annual "
        "values for both NetIncomeLoss and Revenues (call get_xbrl_fact for each "
        "concept separately). Summarize the NetIncomeLoss trend in 3 sentences "
        "and flag anything odd about data quality or gaps in the reported values."
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

    write_audit_log(ticker, goal, result)

    # A single, clean message: just the verified research assessment. The
    # verdict, attempt count, tool log, and audit path all live in the JSON
    # audit file for anyone who needs to trace the answer back to its data.
    print(f"\n{result.final_answer}")


if __name__ == "__main__":
    main()
