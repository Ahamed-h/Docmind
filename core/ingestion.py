import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import time
from core.config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    VECTOR_STORE_PATH
)

# Load embedding model once at module level
# Downloads 22MB on first run, cached after
print("Loading embedding model...")
_embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model loaded.")


def load_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        text += f"\n[PAGE {page_num + 1}]\n{page_text}"
    return text


def chunk_text(text: str) -> list[str]:
    """
    Split into overlapping chunks.
    RecursiveCharacterTextSplitter splits at paragraph
    → sentence → word boundaries in that order.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    # Remove empty chunks
    return [c.strip() for c in chunks if c.strip()]


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """
    Embed all chunks using local sentence-transformers.
    No API calls. No timeouts. No cost.
    batch_size=32 is optimal for CPU inference.
    """
    print(f"  Embedding {len(chunks)} chunks locally...")
    embeddings = _embedder.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    vector = _embedder.encode([query], convert_to_numpy=True)
    return vector.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS flat L2 index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_index(index, chunks: list[str]):
    """Save FAISS index and chunks to disk."""
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    faiss.write_index(index, f"{VECTOR_STORE_PATH}/index.faiss")
    with open(f"{VECTOR_STORE_PATH}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {VECTOR_STORE_PATH}")


def load_index():
    """Load FAISS index and chunks from disk."""
    index = faiss.read_index(f"{VECTOR_STORE_PATH}/index.faiss")
    with open(f"{VECTOR_STORE_PATH}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def ingest_folder(folder_path: str):
    """Ingest all PDFs from folder."""
    all_chunks = []

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"No PDF files found in {folder_path}")

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        path = os.path.join(folder_path, pdf_file)
        print(f"Processing: {pdf_file}")
        text = load_pdf(path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks extracted")

    print(f"\nTotal chunks: {len(all_chunks)}")

    embeddings = embed_chunks(all_chunks)
    index = build_faiss_index(embeddings)
    save_index(index, all_chunks)

    print("Index built successfully.")
    return len(all_chunks)