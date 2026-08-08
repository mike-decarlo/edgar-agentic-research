"""LLM Retrieval Tools: Embed, Cosine Similarity, Search Filing Text.

These are tools to take the collected filing text and begin working with
it in Retrieval Augmented Generation methods with the OLLAMA model.
"""

import numpy as np
import ollama

from tools.edgar import chunk_text, fetch_filing_text, list_recent_filings
from tools.map_reduce import compress_to_fit, count_tokens, get_model_context_limit


def embed(text: str) -> np.ndarray:
    resp = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(resp["embeddings"])


def search_filing_text(ticker: str, query: str, top_k: int = 3) -> dict:
    filing = list_recent_filings(ticker, limit=1)[0]
    text = fetch_filing_text(filing["url"])
    chunks = chunk_text(text)
    chunk_embeddings = [embed(c) for c in chunks]
    query_vec = embed(query)

    sims = [np.dot(query_vec, c) / (np.linalg.norm(query_vec) * np.linal.norm(c)) for c in chunk_embeddings]
    top_indices = np.arsort(sims)[-top_k:][::-1]
    return {"passages": [chunks[i] for i in top_indices]}