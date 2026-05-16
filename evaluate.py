"""
Run this after build_index.py to get RAGAS scores.
Usage: python evaluate.py
"""

from core.ingestion import load_index
from core.retrieval import retrieve
from core.generation import generate, rewrite_query
from core.evaluation import run_ragas_evaluation


TEST_CASES = [
    {
        "question": "How long is maternity leave?",
        "ground_truth": "Eligible employees can avail paid maternity leave for a continuous period of 26 weeks."
    },
    {
        "question": "What happens if an employee violates the code of conduct?",
        "ground_truth": "Failure to adhere to the code could attract severe consequences including termination of employment."
    },
    {
        "question": "Can employees accept gifts from suppliers?",
        "ground_truth": "Employees should not solicit business courtesies or gifts. Gifts cumulatively valued at Rs.5000 or less during the calendar year are considered nominal."
    },
    {
        "question": "What is the probationary period for new employees?",
        "ground_truth": "Newly hired employees are in a probationary period for 6 months from the date of hire."
    },
    {
        "question": "What is the probationary period for new employees?",
        "ground_truth": "Newly hired employees are in a probationary period for 6 months from the date of hire."
    },
    {
        "question": "What is the company policy on tobacco use?",
        "ground_truth": "Employees may only smoke or chew tobacco during authorised breaks and must avoid smoking in front of superiors, customers or vendors."
    }
]


def main():
    print("Loading index...")
    index, chunks = load_index()
    print(f"Loaded {len(chunks)} chunks\n")

    print("Running RAG pipeline on test questions...")
    evaluated_cases = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"Question {i}/{len(TEST_CASES)}: {case['question']}")

        rewritten = rewrite_query(case["question"])
        top_chunks = retrieve(rewritten, index, chunks)
        result = generate(rewritten, top_chunks)

        retrieved_contexts = [
            s["text"] if isinstance(s, dict) and "text" in s else str(s)
            for s in result["sources"]
        ]

        evaluated_cases.append({
            "user_input": case["question"],
            "response": result["answer"],
            "retrieved_contexts": retrieved_contexts,
            "reference": case["ground_truth"],
        })

        print(f"  Answer: {result['answer'][:80]}...")
        print(f"  Confidence: {result['confidence']}\n")

    print("Running RAGAS evaluation...")
    scores = run_ragas_evaluation(evaluated_cases)

    print("\n" + "=" * 40)
    print("RAGAS SCORES")
    print("=" * 40)

    if "faithfulness" in scores:
        print(f"Faithfulness:      {scores['faithfulness']}")
    if "context_precision" in scores:
        print(f"Context Precision: {scores['context_precision']}")
    if "context_recall" in scores:
        print(f"Context Recall:    {scores['context_recall']}")
    if "answer_relevancy" in scores:
        print(f"Answer Relevancy:  {scores['answer_relevancy']}")
    if "overall_score" in scores:
        print(f"Overall Score:     {scores['overall_score']}")

    print("=" * 40)
    print("\nPaste these into your README and sidebar.")
if __name__ == "__main__":
    main()