"""Streamlit front end for the EDGAR researcher -> verifier pipeline.

A thin wrapper around ``run_pipeline_with_retry`` so a reviewer can drive the
system from a browser and *see the machinery*: the researcher's ReAct tool
trace, the deterministic numeric sanity gate, and the LLM verifier's verdict —
not just a final sentence.

Secrets (the hosted-model API key, the SEC User-Agent) come from Streamlit's
secrets manager, never from committed code. Locally that's
``.streamlit/secrets.toml`` (git-ignored); on Streamlit Community Cloud it's the
app's Secrets box. We copy them into the process environment so ``config.py`` and
``llm.py`` — which read ``os.environ`` and know nothing about Streamlit — work
unchanged in either place.
"""

import os

import streamlit as st

# --- Wire Streamlit secrets into the environment BEFORE importing the pipeline,
# --- since config.py reads these at import time.
_SECRET_KEYS = ("SEC_USER_AGENT", "GROQ_API_KEY", "LLM_PROVIDER", "GROQ_MODEL", "OLLAMA_MODEL")
try:
    for _key in _SECRET_KEYS:
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:  # noqa: BLE001 - no secrets.toml locally is fine; env may be set another way
    pass

from agents.verifier import run_pipeline_with_retry  # noqa: E402
from llm import active_model, current_provider  # noqa: E402
from tools.edgar import TickerNotFoundError  # noqa: E402

DEFAULT_QUESTION = (
    "Summarize the NetIncomeLoss trend in 2-3 paragraphs and flag anything odd "
    "about data quality or gaps in the reported values."
)


def build_goal(ticker: str, question: str) -> str:
    """Turn a ticker + free-text question into a tool-driving research goal."""
    return (
        f"Look up {ticker}'s most recent 10-K filing. Pull the historical annual "
        "values for both NetIncomeLoss and Revenues (call get_xbrl_fact for each "
        f"concept separately). {question}"
    )


st.set_page_config(page_title="EDGAR Agentic Research", page_icon="📑", layout="wide")

# Constrain the main content to 70% of the viewport width and center it. Streamlit's
# built-in "centered" layout uses a fixed pixel max-width; we want a responsive 70%,
# so we keep the "wide" layout and narrow the main block container with CSS instead.
st.markdown(
    """
    <style>
      .block-container {
        max-width: 70%;
        margin-left: auto;
        margin-right: auto;
        padding-left: 1rem;
        padding-right: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📑 EDGAR Agentic Research")
st.caption(
    "A researcher agent pulls real SEC EDGAR data via a ReAct tool loop; a "
    "deterministic sanity gate and an LLM verifier then check its work before "
    "you see an answer."
)

# Show which backend is live so a reviewer knows local Ollama vs. hosted fallback.
try:
    _provider = current_provider()
    _badge = "🖥️ local Ollama" if _provider == "ollama" else "☁️ hosted (Groq)"
    st.info(f"Active model backend: **{_badge}** — `{active_model()}`")
except Exception as exc:  # noqa: BLE001
    st.error(f"No LLM backend configured: {exc}")

with st.form("research"):
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("Company ticker", value="AAPL").strip().upper()
    with col2:
        question = st.text_area("Research question", value=DEFAULT_QUESTION, height=80).strip()
    submitted = st.form_submit_button("Run research", type="primary")

if submitted:
    if not ticker:
        st.warning("Enter a ticker to research.")
        st.stop()

    goal = build_goal(ticker, question or DEFAULT_QUESTION)

    with st.spinner(f"Researching {ticker} — running the ReAct loop and checks…"):
        try:
            result = run_pipeline_with_retry(goal)
        except TickerNotFoundError as exc:
            st.error(f"Could not research **{ticker}**: {exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001 - surface network/API failures readably
            st.exception(exc)
            st.stop()

    # --- Verdict header ----------------------------------------------------
    if result.passed:
        st.success(f"✅ Verified — passed in {result.attempts} attempt(s)")
    elif result.numeric_issues:
        st.error("⛔ Stopped by the deterministic numeric sanity gate (no LLM verifier call)")
    else:
        st.warning(f"⚠️ Not verified after {result.attempts} attempt(s) — see verifier issues below")

    # --- Final answer ------------------------------------------------------
    st.subheader("Final answer")
    st.write(result.final_answer)

    # --- Deterministic sanity gate ----------------------------------------
    st.subheader("Deterministic numeric sanity gate")
    if result.numeric_issues:
        for issue in result.numeric_issues:
            st.error(issue)
    else:
        st.success("No magnitude/accounting-identity problems found in the raw XBRL data.")

    # --- LLM verifier ------------------------------------------------------
    st.subheader("LLM verifier (fact-check vs. raw data)")
    if result.verifier_check is not None:
        verdict = "valid" if result.verifier_check.get("valid") else "invalid"
        st.write(f"Verdict: **{verdict}**")
        issues = result.verifier_check.get("issues") or []
        if issues:
            for issue in issues:
                st.write(f"- {issue}")
        st.json(result.verifier_check)
    else:
        st.caption("Verifier was skipped (the sanity gate fired first).")

    # --- ReAct tool trace --------------------------------------------------
    st.subheader(f"ReAct tool trace ({len(result.tool_log)} tool call(s))")
    if not result.tool_log:
        st.caption("The researcher answered without calling any tools.")
    for i, entry in enumerate(result.tool_log, start=1):
        with st.expander(f"{i}. {entry['tool']}({entry.get('args', {})})"):
            st.json(entry["result"])
