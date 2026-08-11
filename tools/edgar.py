"""SEC EDGAR data-access tools: ticker->CIK, recent filings, and XBRL facts.

These are plain functions with no LLM dependency. Keeping tools separate from
the agent loop is the whole point: swap the model, keep the tools. Each function
hits a public SEC EDGAR endpoint and returns plain dicts/lists ready to hand
back to a model as a tool result.
"""

import logging
from datetime import date

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
    limit = int(limit)  # models sometimes hand us "3" as a string
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
    """Keep only facts whose own start/end span is genuinely ~1 year, since
    filers sometimes mistag a shorter duration as fp: "FY". This is about
    correctly identifying *annual* facts, not about requiring consecutive years.
    """
    annual = []
    for v in values:
        if v.get("form") != "10-K":
            continue
        start, end = v.get("start"), v.get("end")
        if not start or not end:
            continue
        duration_days = (date.fromisoformat(end) - date.fromisoformat(start)).days
        if 330 <= duration_days <= 400:
            annual.append(v)

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