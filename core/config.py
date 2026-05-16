from dotenv import load_dotenv
import os

load_dotenv()

# Embedding — local model, no API needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 22MB, runs on CPU, fast
EMBEDDING_DIMENSION = 384              # dimension for this model

# LLM — still uses Gemini API for generation only
LLM_MODEL = "models/gemini-2.5-flash"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval
FAISS_TOP_K = 20
BM25_TOP_K = 20
RERANK_TOP_N = 3
CONFIDENCE_THRESHOLD = 0.6

# Paths
VECTOR_STORE_PATH = "vector_store/faiss_index"
DATA_PATH = "data/sample_docs"

# API Keys — only needed for generation now
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")