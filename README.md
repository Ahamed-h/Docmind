# 📄 DocMind — Enterprise Document Intelligence

> A production-grade retrieval-augmented generation (RAG) system that answers questions from enterprise documents with grounded citations, low hallucination risk, and measurable evaluation.

**Live Demo:** Under deployment  
**GitHub:** [Ahamed-h/DocMind](https://github.com/Ahamed-h/DocMind)

***

## The Problem

Organizations store critical knowledge inside PDFs such as HR policies, legal documents, compliance manuals, and SOPs. Manually searching these files is slow, and generic LLMs can hallucinate answers from pretraining instead of using the actual document set.

DocMind addresses this by retrieving relevant passages from uploaded documents, generating an answer grounded in those passages, attaching source citations, and returning an abstaining response when the required information is not present.

***

## RAGAS Evaluation Results

Evaluated on a 6-question test set using Gemini as the judge model and RAGAS metrics.

| Metric | Score | Interpretation |
|--------|------:|----------------|
| **Faithfulness** | **1.000** | Every evaluated claim was supported by retrieved context. |
| **Context Precision** | **0.750** | Most retrieved chunks were relevant, though some retrieval noise remains. |
| **Context Recall** | **0.833** | Most of the information needed to answer the query was retrieved. |
| **Answer Relevancy** | **0.797** | Answers generally addressed the user question well. |
| **Overall Score** | **0.845** | Strong end-to-end baseline across grounding, retrieval, and answer quality. |

A faithfulness score of 1.0 is especially important because it indicates the system did not add unsupported claims in the evaluated outputs.

***

## Architecture

```text
User Query
   │
   ▼
Query Rewriting (Gemini)
   │  Rewrites vague queries for better retrieval
   ▼
┌──────────────────────────────────────────────┐
│               Hybrid Retrieval               │
│                                              │
│   FAISS (semantic) ──┐                       │
│                      ├── RRF ──► Top merged  │
│   BM25 (keyword) ────┘                       │
└──────────────────────────────────────────────┘
   │
   ▼
Cross-Encoder Reranking
   │  Selects the most relevant chunks
   ▼
Gemini Generation
   │
   ▼
LLM-as-Judge Hallucination Check
   │  Confidence score from 0 to 1
   │  Below threshold → "Insufficient information"
   ▼
Answer + Confidence + Source Citations
```

***

## Why This Stack

### FAISS over Pinecone

FAISS runs locally with no API cost or network dependency. For hundreds to low thousands of chunks, a local FAISS index is fast and simple to manage, while Pinecone is more useful at much larger operational scale.

### Local embeddings over API embeddings

Local sentence-transformer embeddings avoid API quotas, timeout issues, and repeated inference costs. This also makes indexing more reliable and reproducible.

### BM25 plus FAISS over FAISS alone

Hybrid retrieval improves robustness. BM25 captures exact keyword and section-number matches, while FAISS captures semantic similarity.

### Cross-encoder reranking

Initial retrieval surfaces candidates; reranking improves the final context set by scoring query-chunk relevance more accurately than raw vector similarity alone.

***

## Features

- Upload and query PDF document collections.
- Hybrid retrieval with FAISS semantic search and BM25 keyword search.
- Reciprocal Rank Fusion (RRF) for ranked-list merging.
- Cross-encoder reranking for better context precision.
- Query rewriting before retrieval.
- LLM-as-judge hallucination detection.
- Confidence score with every answer.
- Source passage citations in responses.
- Graceful fallback when information is missing.
- Session chat history.
- RAGAS-based evaluation pipeline.

***

## Tech Stack

| Layer | Tool |
|------|------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (`IndexFlatL2`) |
| Keyword Search | BM25 (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Retrieval Merge | Reciprocal Rank Fusion |
| LLM | Gemini Flash |
| Evaluation | RAGAS |
| PDF Parsing | PyMuPDF |
| Frontend | Streamlit |
| Language | Python 3.12 |

***

## Project Structure

```text
docmind/
├── core/
│   ├── config.py          # Centralized settings
│   ├── ingestion.py       # PDF loading, chunking, embeddings, FAISS build
│   ├── retrieval.py       # Hybrid retrieval, RRF merge, reranking
│   ├── generation.py      # Generation, query rewriting, confidence checks
│   └── evaluation.py      # RAGAS evaluation logic
├── data/
│   └── sample_docs/       # Input PDFs
├── vector_store/          # Local FAISS index (gitignored)
├── streamlit_app.py       # Streamlit app entry point
├── build_index.py         # Build index from documents
├── evaluate.py            # Run evaluation metrics
├── requirements.txt
├── .env.example
├── DECISIONS.md
└── README.md
```

***

## Run Locally

### 1. Clone the repository and create a virtual environment

```bash
git clone https://github.com/Ahamed-h/DocMind.git
cd DocMind
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` and add the required API keys.

### 4. Add your PDFs

Place PDF files inside `data/sample_docs/`.

### 5. Build the index

```bash
python build_index.py
```

### 6. Run the application

```bash
streamlit run streamlit_app.py
```

***

## Run Evaluation

```bash
python evaluate.py
```

Update the test questions in `evaluate.py` so they match the content of your documents. The evaluation scores are printed in the terminal.

***

## Example Questions

Questions suited to the included HR policy documents:

- How long is maternity leave?
- What happens if an employee violates the code of conduct?
- Can employees accept gifts from suppliers?
- What is the probationary period for new employees?
- What is the company policy on tobacco use?

Out-of-scope examples that should trigger an abstaining response:

- What is the company's stock price?
- Who is the CEO of Apple?

***

## Errors Faced and Fixes

### 1. Gemini embedding API timeouts

Large embedding batches caused timeout failures during development. This was resolved by switching to local sentence-transformer embeddings, which removed API dependence from the indexing path.

### 2. Gemini model-name mismatch

An earlier Gemini model name did not match the installed SDK behavior. The model configuration was corrected after checking available model naming in the active client setup.

### 3. RAGAS embedding interface mismatch

`answer_relevancy` failed because the Google embedding object exposed `embed_text()` while RAGAS expected an interface compatible with `embed_query()`. This was fixed by using Hugging Face embeddings through a compatible wrapper.

### 4. RAGAS result aggregation issue

RAGAS returned per-row metric lists rather than a single scalar. The reporting logic was updated to average the metric values safely before printing.

### 5. Missing metric print keys

After temporarily removing some metrics during debugging, the report code still tried to print absent keys. This was fixed by printing metrics conditionally only when present.

### 6. Context recall setup

`context_recall` required a `reference` field in each evaluation sample. The evaluation dataset was updated to include ground-truth answers for every test case.

### 7. OpenAI dependency issue

Using OpenAI embeddings was dropped because of API limits. The final setup used Hugging Face embeddings locally for `answer_relevancy` instead.

***

## What Could Be Improved

- Persistent storage for query and evaluation logs.
- Token streaming in the Streamlit interface.
- Better source attribution across multiple PDFs.
- Automatic index rebuilds when documents change.
- A larger evaluation set for more robust benchmarking.
- Metadata-based filtering by document name or category.

***

## Technical Decisions

See [DECISIONS.md](./DECISIONS.md) for a full decision log covering FAISS vs. managed vector databases, local vs. API embeddings, hybrid retrieval, reranking, model choices, and evaluation design.
