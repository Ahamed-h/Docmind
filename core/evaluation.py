import os
from datasets import Dataset
from google import genai

from ragas import evaluate
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
)

from langchain_huggingface import HuggingFaceEmbeddings

from core.config import GOOGLE_API_KEY


def get_ragas_config():
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    client = genai.Client(api_key=GOOGLE_API_KEY)
    cache = DiskCacheBackend(cache_dir=".cache")

    ragas_llm = llm_factory(
        "gemini-2.5-flash",
        provider="google",
        client=client,
        cache=cache,
    )

    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        embeddings=hf_embeddings,
        cache=cache,
    )

    return ragas_llm, ragas_embeddings


def mean_safe(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def run_ragas_evaluation(test_cases: list[dict]) -> dict:
    ragas_llm, ragas_embeddings = get_ragas_config()
    dataset = Dataset.from_list(test_cases)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=True,
    )

    scores = {
        "faithfulness": mean_safe(results["faithfulness"]),
        "context_precision": mean_safe(results["context_precision"]),
        "context_recall": mean_safe(results["context_recall"]),
        "answer_relevancy": mean_safe(results["answer_relevancy"]),
    }

    scores["overall_score"] = round(sum(scores.values()) / len(scores), 3)
    return scores