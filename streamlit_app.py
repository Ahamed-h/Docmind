import streamlit as st
import os
from core.ingestion import load_index
from core.retrieval import retrieve
from core.generation import generate, rewrite_query
from core.config import VECTOR_STORE_PATH

st.set_page_config(
    page_title="DocMind — Document Intelligence",
    page_icon="📄",
    layout="wide"
)

# ── Load index once using cache ───────────────────────────────
@st.cache_resource
def load_rag_index():
    """Load FAISS index and chunks. Runs once, cached after."""
    faiss_path = f"{VECTOR_STORE_PATH}/index.faiss"
    if not os.path.exists(faiss_path):
        return None, None
    try:
        return load_index()
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return None, None

# ── Header ────────────────────────────────────────────────────
st.title("📄 DocMind — Enterprise Document Intelligence")
st.caption(
    "Ask questions about your documents. "
    "Every answer is grounded in source — no hallucinations."
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("How It Works")
    st.write("""
    **Retrieval Pipeline:**
    1. Query rewriting — improves search
    2. FAISS semantic search
    3. BM25 keyword search
    4. RRF merge — combines both
    5. Cross-encoder reranking

    **Generation Pipeline:**
    6. Gemini 1.5 Flash generates answer
    7. LLM-as-judge hallucination check
    8. Confidence scoring
    9. Source citations returned
    """)

    st.divider()
    st.subheader("RAGAS Evaluation")
    st.caption("Measured on 10-question test set")
    col1, col2 = st.columns(2)
    col1.metric("Faithfulness", "—")
    col1.metric("Relevancy", "—")
    col2.metric("Precision", "—")
    st.caption("Run evaluate.py to get scores")

# ── Load index ────────────────────────────────────────────────
index, chunks = load_rag_index()

if index is None:
    st.warning(
        "No document index found. "
        "Add PDFs to data/sample_docs/ and run: "
        "`python build_index.py`"
    )
    st.stop()

st.sidebar.success(f"✅ {len(chunks)} chunks indexed")

# ── Chat history ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            c1, c2 = st.columns(2)
            c1.caption(f"Confidence: {msg.get('confidence', '—')}")
            c2.caption(
                "✅ Grounded" if not msg.get("hallucination_detected")
                else "⚠️ Check answer"
            )
            if msg.get("rewritten_query"):
                st.caption(f"Search query used: _{msg['rewritten_query']}_")

# ── Query input ───────────────────────────────────────────────
query = st.chat_input("Ask a question about your documents...")

if query:
    # Show user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("assistant"):

        # Step 1 — Rewrite query
        with st.spinner("Understanding your question..."):
            rewritten = rewrite_query(query)

        # Show rewritten query if different
        if rewritten.lower() != query.lower():
            st.caption(f"🔍 Searching for: _{rewritten}_")

        # Step 2 — Retrieve
        with st.spinner("Searching documents..."):
            top_chunks = retrieve(rewritten, index, chunks)

        # Step 3 — Generate
        with st.spinner("Generating answer..."):
            result = generate(rewritten, top_chunks)

        # Show answer
        st.write(result["answer"])

        # Show metadata
        c1, c2 = st.columns(2)
        c1.caption(f"Confidence: {result['confidence']}")
        c2.caption(
            "✅ Grounded" if not result["hallucination_detected"]
            else "⚠️ Possible hallucination — verify manually"
        )

        # Show sources
        with st.expander("📄 Source passages used"):
            for i, src in enumerate(result["sources"], 1):
                st.markdown(f"**Passage {i}:**")
                st.text(src[:400] + "..." if len(src) > 400 else src)
                st.divider()

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "confidence": result["confidence"],
            "hallucination_detected": result["hallucination_detected"],
            "rewritten_query": rewritten
        })