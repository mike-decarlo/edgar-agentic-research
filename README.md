# EDGAR Agentic Research

A small, framework-free **multi-agent** system that researches a public
company's SEC filings and then **checks its own work** before returning an
answer. It runs entirely against **local Ollama models** (default
`qwen2.5:14b`) — no API keys, no cloud, no per-token cost.

Given a ticker, a **researcher** agent pulls real data from the SEC EDGAR API
(recent filings and XBRL financial facts), summarizes the trend, and flags data
oddities. A **verifier** stage then confirms that every claim in the summary is
actually supported by the raw data the researcher collected.

---

## Why researcher + verifier?

A single LLM asked to "look this up and summarize it" will confidently invent
numbers, drop caveats, and occasionally answer in the wrong language. Splitting
the work into two narrow roles makes each one easier to trust:

- **Researcher** — has tools, does the lookups, produces a summary *and* returns
  a `tool_log`: the exact tool calls, arguments, and raw results it saw.
- **Verifier** — has **no tools**. Its only job is to compare the researcher's
  prose against that `tool_log` and flag anything unsupported. It can't wander
  off and do new research; it can only check.

If the verifier finds problems, the orchestrator feeds them back to the
researcher as correction instructions and retries.

## Architecture: two layers of checking, cheapest first

```
ticker ─▶ Researcher (ReAct tool loop) ─▶ answer + tool_log
                                              │
                                              ▼
                    1. check_numeric_sanity  ── deterministic, NO LLM
                       (arithmetic magnitude gate)
                                              │
                          passes? ──no──▶ stop, report data error
                                              │ yes
                                              ▼
                    2. verify_answer         ── LLM, NO tools
                       (semantic fact-check vs raw data)
                                              │
                          valid? ──no──▶ feed issues back, retry
                                              │ yes
                                              ▼
                                        final answer + audit log
```

### Why the deterministic gate runs *before* the LLM verifier

**Code should own arithmetic correctness; LLMs should own semantic and
reasoning checks.** LLMs are unreliable at eyeballing large raw numbers in JSON
— they'll miss that `$96,990,000,000` and `$9,370,000,000` are an order of
magnitude apart. So `check_numeric_sanity` is plain Python: it walks the
researcher's XBRL results and flags any year-over-year move below ~⅓ or above 3×,
which for a stable balance-sheet concept almost always signals a data or scale
error rather than real change.

This runs as a **gate**: if the numbers are already impossible, we stop and
report a data error *without* spending an LLM call — a magnitude error is
something the verifier can't reason its way out of anyway. Only if the
arithmetic is sane do we spend the LLM verifier on the harder, fuzzier question:
"is the *prose* faithful to the data, and are the right caveats present?"

### The dedup bug this was built to catch

The SEC `companyconcept` API returns **every** time a value was reported for a
concept — including the comparative prior-year columns that later filings
restate. So a single fiscal year can appear multiple times with different filing
dates and, occasionally, different values. Without deduping, a later 10-K's
comparative column collided with the original entry and the agent reported
**two different values ($96.99B and $9.37B) for the same fiscal year**.

`tools/edgar._select_annual_facts` fixes this: keep only annual (`10-K` / `FY`)
figures, then for each fiscal-year end keep the **most recently filed** entry.
There's a dedicated regression test for exactly this scenario in
`tests/test_edgar.py`.

---

## Project structure

```
edgar-agentic-research/
├── config.py            # model + SEC User-Agent (from environment)
├── tools/
│   ├── edgar.py         # get_cik, list_recent_filings, get_xbrl_fact, tool schemas
│   ├── map_reduce.py    # controlled map reduction for text that will overflow context window on retrieval
|   └── retrieval.py     # RAG for relevance-based retreival
├── agents/
│   ├── researcher.py    # run_researcher — the ReAct tool loop
│   └── verifier.py      # check_numeric_sanity, verify_answer, run_pipeline_with_retry
├── audit.py             # writes each run to logs/<timestamp>_<TICKER>.json
├── main.py              # CLI entry point
├── tests/               # offline unit + regression tests (mocked SEC API)
│   ├── test_edgar.py    # tests for edgar tools like cik resolution and deduplication by fiscal year
│   ├── test_sanity.py   # tests for numeric sanity checking of ratio thresholds
├── requirements.txt     # runtime deps (pinned)
├── requirements-dev.txt # + pytest
└── .env.example         # SEC User-Agent template
```

---

## Setup

### 1. Install and start Ollama

Install Ollama from <https://ollama.com/download>, then pull a tool-capable
model:

```bash
ollama pull qwen2.5:14b
```

Ollama runs as a local server; make sure it's running (`ollama list` should
work) before starting the agent.

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure your SEC User-Agent

The SEC EDGAR API **requires** every request to identify who's making it (a name
and contact email). Copy the example env file and fill it in:

```bash
cp .env.example .env
# then edit .env and set, e.g.:
# SEC_USER_AGENT=Jane Doe jane@example.com
```

No real email is committed to the repo — `.env` is gitignored; only
`.env.example` is tracked.

### 4. Run

```bash
python main.py
# Enter a stock ticker (e.g. AAPL, GOOGL): AAPL
```

Each run prints the final answer and writes a full audit record (goal, answer,
verifier verdict, and the complete raw tool log) to `logs/`.

---

## Running the tests

The tests mock every SEC API call, so they're fully offline and fast.

```bash
pip install -r requirements-dev.txt
pytest
# or, with no extra dependencies:
python -m unittest discover -s tests
```

They cover the fiscal-year dedup logic, the `check_numeric_sanity` ratio
thresholds (including exact boundary cases), and a regression test for the
$96.99B/$9.37B duplicate-fiscal-year bug described above.

---

## Example run

A real run against `AAPL` (`qwen2.5:14b`):

```text
$ python main.py
Enter a stock ticker (e.g. AAPL, GOOGL): AAPL

Apple Inc.'s (AAPL) NetIncomeLoss values were consistently reported in recent
filings: $99.8 billion for fiscal year 2022, $96.9 billion for fiscal year 2023,
$93.7 billion for fiscal year 2024, and a recovery to $112 billion for fiscal
year 2025. The values for fiscal years 2024 and 2025 were reported on the same
filing date (October 31, 2025), reflecting that later years' figures come from
the most recent 10-K.
```

The deduped data those figures are drawn from (one row per fiscal year, most
recently filed value):

| Fiscal year end | NetIncomeLoss  | Filed      |
|-----------------|----------------|------------|
| 2022-09-24      | $99.803B       | 2024-11-01 |
| 2023-09-30      | $96.995B       | 2025-10-31 |
| 2024-09-28      | $93.736B       | 2025-10-31 |
| 2025-09-27      | $112.010B      | 2025-10-31 |

**What this run shows.** The final figures match the SEC data exactly, and the
fiscal-year dedup did its job — each year appears once.

---

## Notes & limitations

- Output quality depends on the local model; `qwen2.5:14b` is a reasonable
  balance of tool-use reliability and size. Override with `OLLAMA_MODEL`.
- Tickers that don't file with the SEC (private companies, funds/ETFs, most
  non-US listings) are reported cleanly rather than crashing.
- This is a teaching-sized system on purpose — two roles, one shared context, no
  agent framework — meant to show the researcher/verifier pattern and the
  code-owns-arithmetic / LLM-owns-semantics split clearly.
