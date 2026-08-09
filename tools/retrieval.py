"""LLM Retrieval Tools: Embed, Cosine Similarity, Search Filing Text.

These are tools to take the collected filing text and begin working with
it in Retrieval Augmented Generation methods with the OLLAMA model.
"""

import numpy as np
import ollama

from tools.edgar import chunk_text, fetch_filing_text, list_recent_filings
from tools.map_reduce import (
    CHAT_MODEL,
    compress_to_fit,
    count_tokens,
    get_model_context_limit,
)


def embed(text: str) -> np.ndarray:
    resp = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(resp["embeddings"][0])


def search_filing_text(ticker: str, query: str, top_k: int = 3) -> dict:
    filing = list_recent_filings(ticker, limit=1)[0]
    text = fetch_filing_text(filing["url"])

    # Gap 1: raw 10-Ks can run 40K-80K+ tokens -- compress before indexing
    limit = get_model_context_limit(CHAT_MODEL)
    if count_tokens(text) > limit * 0.5: # rough "worth compressing" threshold
        text = compress_to_fit(text, goal=query, model=CHAT_MODEL)

    chunks = chunk_text(text)
    chunk_embeddings = [embed(c) for c in chunks]
    query_vec = embed(query)

    sims = [np.dot(query_vec, c) / (np.linalg.norm(query_vec) * np.linalg.norm(c)) for c in chunk_embeddings]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    retrieved = [chunks[i] for i in top_indices]

    combined = "\n\n".join(retrieved)

    # Gap 2: this is the problem we actually hit -- top-k chunks combined
    # can still exceed the budget even after pre-compression
    if count_tokens(combined) > limit * 0.85:
        combined = compress_to_fit(retrieved, goal=query, model=CHAT_MODEL)

    return {"passages": combined}