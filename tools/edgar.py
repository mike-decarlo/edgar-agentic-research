"""SEC EDGAR data-access tools: ticker->CIK, recent filings, and XBRL facts.

These are plain functions with no LLM dependency. Keeping tools separate from
the agent loop is the whole point: swap the model, keep the tools. Each function
hits a public SEC EDGAR endpoint and returns plain dicts/lists ready to hand
back to a model as a tool result.
"""

import logging

import requests
from bs4 import BeautifulSoup

from config import get_headers

logger = logging.getLogger(__name__)


class TickerNotFoundError(ValueError):
    """Raised when a ticker has no matching CIK in SEC's ticker map.

    Typically means the ticker is a private company, an ETF/fund, or a
    non-US listing that does not file with the SEC.
    """


def get_cik(ticker: str) -> str:
    """Resolve a ticker to a 10-digit zero-padded CIK using SEC's ticker map."""
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=get_headers())
    resp.raise_for_status()
    data = resp.json()
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise TickerNotFoundError(
        f"Ticker {ticker!r} was not found in SEC's ticker map. It may be a "
        "private company, a fund/ETF, or a non-US listing that does not file "
        "with the SEC."
    )


def list_recent_filings(
    ticker: str, form_type: str = "10-K", limit: int = 3
) -> list[dict]:
    """Pull recent filings of a given form type for a ticker."""
    cik = get_cik(ticker)
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=get_headers())
    resp.raise_for_status()
    data = resp.json()
    recent = data["filings"]["recent"]
    out = []
    for form, date, acc, doc in zip(
        recent["form"],
        recent["filingDate"],
        recent["accessionNumber"],
        recent["primaryDocument"],
    ):
        if form == form_type:
            acc_nodash = acc.replace("-", "")
            out.append(
                {
                    "form": form,
                    "date": date,
                    "accession": acc,
                    "url": (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{int(cik)}/{acc_nodash}/{doc}"
                    ),
                }
            )
        if len(out) >= limit:
            break
    return out


def fetch_filing_text(url: str) -> str:
    resp = requests.get(url, headers=get_headers())
    # 10-Ks are HTML - strip tags, keep readable text
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def _select_annual_facts(values: list[dict], limit: int = 4) -> list[dict]:
    """Reduce raw companyconcept unit entries to one row per fiscal year.

    The SEC companyconcept API returns *every* time a value was reported for a
    concept — including the comparative prior-year columns that later filings
    restate. So a single fiscal-year end can appear multiple times with
    different ``filed`` dates (and occasionally different ``val``s). Without
    deduping, a later 10-K's comparative column collides with the original
    entry and you can end up reporting two very different numbers for the *same*
    fiscal year (the real bug that once surfaced $96.99B vs $9.37B for one year).

    We keep only annual figures (``form == "10-K"`` and ``fp == "FY"``), then for
    each fiscal-year end keep the most recently *filed* entry, and return the
    last ``limit`` years sorted by fiscal-year end.
    """
    annual = [v for v in values if v.get("form") == "10-K" and v.get("fp") == "FY"]

    by_end: dict[str, dict] = {}
    for v in annual:
        end = v["end"]
        if end not in by_end or v["filed"] > by_end[end]["filed"]:
            by_end[end] = v

    return sorted(by_end.values(), key=lambda v: v["end"])[-limit:]


def _trim_fact(v: dict) -> dict:
    """Project a raw entry down to the fields a model should reason about.

    The raw companyconcept entries carry ``fy`` and ``frame`` fields keyed to the
    *filing* the value appeared in, not the period it describes — e.g. the FY
    ending 2024-09-28 is tagged ``fy: 2025`` because it was restated in the 2025
    10-K. Feeding those back confuses the model into mislabelling years, so we
    drop them and derive ``fiscal_year`` from the period end date itself.
    """
    return {
        "fiscal_year": int(v["end"][:4]),  # calendar year of the period end
        "end": v["end"],
        "val": v["val"],
        "filed": v["filed"],
    }


def get_xbrl_fact(ticker: str, concept: str = "Assets") -> dict:
    """Pull a us-gaap XBRL concept's reported annual values over time.

    Common concepts: Assets, Liabilities, Revenues, NetIncomeLoss,
    StockholdersEquity.
    """
    cik = get_cik(ticker)
    url = (
        f"https://data.sec.gov/api/xbrl/companyconcept/"
        f"CIK{cik}/us-gaap/{concept}.json"
    )
    resp = requests.get(url, headers=get_headers())
    if resp.status_code != 200:
        # Not every company reports every concept — treat as a soft, in-band
        # result the model can reason about, not a hard exception.
        return {"error": f"Concept {concept!r} not found for {ticker}."}
    data = resp.json()
    units = data.get("units", {})
    values = next(iter(units.values()), [])

    deduped = _select_annual_facts(values)
    return {
        "concept": concept,
        "unit": next(iter(units.keys()), None),
        "values": [_trim_fact(v) for v in deduped],
    }