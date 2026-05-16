import google.generativeai as genai
from core.config import LLM_MODEL, CONFIDENCE_THRESHOLD, GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)


def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are a document assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: "I don't have enough information in the provided documents to answer this."
Do not make up information. Cite which part of the context supports your answer.

Context:
{context}

Question: {query}

Answer:"""

def rewrite_query(original_query: str) -> str:
    """
    Rewrite vague user query into better search query.
    Example: "what about leaves?" →
    "employee annual sick maternity leave policy entitlement"
    """
    model = genai.GenerativeModel(LLM_MODEL)
    prompt = f"""Rewrite this question to improve document search retrieval.
Make it specific. Add relevant keywords. Remove filler words.
Return ONLY the rewritten query. Nothing else.

Original: {original_query}
Rewritten:"""
    try:
        response = model.generate_content(prompt)
        rewritten = response.text.strip()
        return rewritten if rewritten else original_query
    except Exception:
        return original_query  # fallback to original if rewrite fails
    
    
def generate(query: str, context_chunks: list[str]) -> dict:
    """
    Generate answer from context.
    Returns: answer, confidence, sources, hallucination_flag
    """
    if not context_chunks:
        return {
            "answer": "No relevant documents found.",
            "confidence": 0.0,
            "sources": [],
            "hallucination_detected": False
        }

    prompt = build_prompt(query, context_chunks)
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    answer = response.text.strip()

    # Hallucination check
    confidence, hallucination = check_hallucination(
        answer, context_chunks, model
    )

    # If confidence too low → override with safe response
    if confidence < CONFIDENCE_THRESHOLD:
        answer = "I don't have enough information in the provided documents to answer this."

    return {
        "answer": answer,
        "confidence": round(confidence, 3),
        "sources": context_chunks,
        "hallucination_detected": hallucination
    }


def check_hallucination(
    answer: str,
    context_chunks: list[str],
    model
) -> tuple[float, bool]:
    """
    LLM-as-judge: check if every claim in answer
    is supported by the retrieved context.
    Returns: confidence score (0-1), hallucination flag (bool)
    """
    context = "\n\n".join(context_chunks)

    check_prompt = f"""Context:
{context}

Answer:
{answer}

Is every claim in the Answer supported by the Context?
Reply in exactly this format:
SUPPORTED: YES or NO
CONFIDENCE: a number between 0.0 and 1.0
REASON: one sentence explanation"""

    check_response = model.generate_content(check_prompt)
    result = check_response.text.strip()

    try:
        lines = result.split("\n")
        supported = "YES" in lines[0].upper()
        confidence_line = [l for l in lines if "CONFIDENCE" in l.upper()][0]
        confidence = float(confidence_line.split(":")[-1].strip())
        hallucination = not supported
    except Exception:
        # If parsing fails → assume medium confidence, not hallucination
        confidence = 0.7
        hallucination = False

    return confidence, hallucination


def generate_streaming(query: str, context_chunks: list[str]):
    """
    Generator version for Streamlit streaming.
    Yields tokens one by one.
    """
    if not context_chunks:
        yield "No relevant documents found."
        return

    prompt = build_prompt(query, context_chunks)
    model = genai.GenerativeModel(LLM_MODEL)

    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text