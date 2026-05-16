import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from core.ingestion import embed_query  # reuse from ingestion
from core.config import (
    FAISS_TOP_K, BM25_TOP_K, RERANK_TOP_N,
    RERANKER_MODEL
)


def faiss_search(query_vector, index, chunks, k=FAISS_TOP_K):
    """Dense semantic search using FAISS."""
    distances, indices = index.search(query_vector, k)
    return [(idx, chunks[idx]) for idx in indices[0] if idx < len(chunks)]


def bm25_search(query: str, chunks: list[str], k=BM25_TOP_K):
    """Sparse keyword search using BM25."""
    tokenized = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:k]
    return [(idx, chunks[idx]) for idx in top_indices]


def reciprocal_rank_fusion(
    faiss_results: list,
    bm25_results: list,
    k: int = 60
) -> list:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    k=60 is the standard constant from the original RRF paper.
    """
    scores = {}

    for rank, (idx, chunk) in enumerate(faiss_results):
        scores[idx] = scores.get(idx, 0) + 1 / (rank + k)

    for rank, (idx, chunk) in enumerate(bm25_results):
        scores[idx] = scores.get(idx, 0) + 1 / (rank + k)

    sorted_ids = sorted(scores, key=scores.get, reverse=True)

    all_chunks_map = {idx: chunk for idx, chunk in faiss_results}
    all_chunks_map.update({idx: chunk for idx, chunk in bm25_results})

    return [(idx, all_chunks_map[idx]) for idx in sorted_ids]


def rerank(query: str, candidates: list, top_n=RERANK_TOP_N) -> list:
    """Cross-encoder reranker — more accurate than vector similarity."""
    if not candidates:
        return []

    reranker = CrossEncoder(RERANKER_MODEL)
    pairs = [(query, chunk) for _, chunk in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    return [chunk for _, (idx, chunk) in ranked[:top_n]]


def retrieve(query: str, index, chunks: list[str]) -> list[str]:
    """
    Full retrieval pipeline:
    embed → FAISS search → BM25 search → RRF merge → rerank
    """
    query_vector = embed_query(query)

    faiss_results = faiss_search(query_vector, index, chunks)
    bm25_results = bm25_search(query, chunks)
    merged = reciprocal_rank_fusion(faiss_results, bm25_results)
    top_chunks = rerank(query, merged)

    return top_chunks