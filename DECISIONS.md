# DocMind — Technical Decisions

## Why FAISS over Pinecone or ChromaDB

FAISS runs locally with zero external service dependency and works well for small to medium document collections. For hundreds to low thousands of chunks, it provides fast retrieval without adding cloud cost, network latency, or operational overhead.

Pinecone is powerful for large-scale managed vector search, but it is unnecessary for this scope. ChromaDB was considered, but the FAISS-based workflow remained simpler and more predictable for local development and deployment.

## Why Local Embeddings over API Embeddings

The indexing pipeline needs to be reliable, cheap, and repeatable. Local sentence-transformer embeddings avoid API rate limits, quota issues, latency spikes, and embedding failures caused by external service instability.

This design keeps ingestion independent from provider availability and makes rebuilding the index easier during development.

## Why Hybrid Search Instead of Only Vector Search

Vector search is strong for semantic similarity, but it can miss exact identifiers such as policy numbers, section references, or narrow keyword phrases. BM25 complements this by rewarding lexical matches.

Combining FAISS and BM25 with Reciprocal Rank Fusion produces stronger retrieval quality than either method alone.

## Why Cross-Encoder Reranking

Initial retrieval methods are good at finding candidates, but not always at ordering them optimally. A cross-encoder reranker scores the query and chunk together, which gives a more accurate notion of relevance than raw vector distance.

This improves the final context passed to generation and helps raise context precision.

## Why No LangGraph

DocMind follows a mostly linear pipeline:

query → retrieve → rerank → generate → validate → answer

There is no complex tool-routing, multi-agent orchestration, or graph-style branching that would justify LangGraph. Plain Python modules keep the project easier to debug, explain, and maintain.

## Why Gemini Flash

For grounded document QA, retrieval quality matters more than using the largest possible generation model. Gemini Flash offers a good trade-off between latency, accessibility, and answer quality for this use case.

A larger model could increase cost and response time without necessarily improving groundedness.

## Why RAGAS for Evaluation

Manual evaluation is slow and subjective. RAGAS provides structured metrics for core RAG qualities:

- Faithfulness
- Context precision
- Context recall
- Answer relevancy

These metrics make the system easier to benchmark, compare, and improve over time.

## Why Hugging Face Embeddings for Answer Relevancy

During evaluation, `answer_relevancy` required an embedding backend compatible with the interface expected by RAGAS. Local Hugging Face embeddings solved this cleanly and avoided dependence on limited OpenAI API usage.

This also kept the full evaluation stack affordable and reproducible.