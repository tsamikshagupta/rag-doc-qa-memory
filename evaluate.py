"""
SQuAD 2.0 Evaluation Runner for the RAG Document Q&A System.

Downloads 100 examples from the SQuAD 2.0 dev set, ingests their
context paragraphs, runs each question through the Q&A pipeline,
and reports F1, Exact Match, answerable-only F1, and unanswerable accuracy.
"""

import json
import time
import requests
from typing import Optional

from rag_docqa_boilerplate import (
    init_vector_db,
    answer_question,
    SessionMemory,
    LongTermMemory,
    get_embedding,
    f1_score,
    exact_match,
)
import chromadb


SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def download_squad_dev(url: str = SQUAD_DEV_URL) -> dict:
    """
    Download the SQuAD 2.0 dev set JSON.

    Args:
        url: URL to the SQuAD 2.0 dev JSON file.

    Returns:
        Parsed JSON dict.
    """
    print("[eval] Downloading SQuAD 2.0 dev set...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    print("[eval] Download complete.")
    return response.json()


def extract_examples(squad_data: dict, n: int = 100) -> list[dict]:
    """
    Extract n question-answer examples from SQuAD 2.0 data.

    Each example dict contains:
      - context (str): the paragraph text
      - question (str): the question
      - answer (str): the gold answer (empty string if unanswerable)
      - is_impossible (bool): whether the question is unanswerable

    Args:
        squad_data: Parsed SQuAD 2.0 JSON.
        n: Number of examples to extract.

    Returns:
        List of example dicts.
    """
    examples: list[dict] = []

    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                is_impossible = qa.get("is_impossible", False)

                if is_impossible:
                    answer = ""
                else:
                    answers = qa.get("answers", [])
                    answer = answers[0]["text"] if answers else ""

                examples.append({
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "is_impossible": is_impossible,
                })

                if len(examples) >= n:
                    return examples

    return examples


def ingest_contexts(examples: list[dict], collection) -> None:
    """
    Ingest unique context paragraphs into ChromaDB for retrieval.

    Args:
        examples: List of SQuAD examples with 'context' key.
        collection: ChromaDB collection to store chunks in.
    """
    seen_contexts: set[str] = set()
    ingested = 0

    for i, ex in enumerate(examples):
        ctx_hash = hash(ex["context"])
        if ctx_hash in seen_contexts:
            continue
        seen_contexts.add(ctx_hash)

        try:
            embedding = get_embedding(ex["context"])
            chunk_id = f"squad_ctx_{i}"
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[ex["context"]],
                metadatas=[{
                    "source": "squad_dev",
                    "doc_id": f"squad_{i}",
                    "chunk_index": 0,
                    "char_start": 0,
                    "char_end": len(ex["context"]),
                    "page_number": -1,
                    "ingested_at": "2026-01-01T00:00:00",
                    "type": "chunk",
                }],
            )
            ingested += 1

            # Rate limiting: brief pause every 10 ingestions
            if ingested % 10 == 0:
                print(f"[eval] Ingested {ingested} contexts...")
                time.sleep(1)

        except Exception as e:
            print(f"[eval] Failed to ingest context {i}: {e}")

    print(f"[eval] Total contexts ingested: {ingested}")


def run_evaluation(examples: list[dict], collection, session: SessionMemory) -> dict:
    """
    Run evaluation on SQuAD examples and compute metrics.

    Computes:
      - Overall mean F1
      - Overall Exact Match rate
      - Answerable-only F1
      - Unanswerable accuracy (% of unanswerable questions correctly
        identified as unanswerable)

    Args:
        examples: List of SQuAD example dicts.
        collection: ChromaDB collection with ingested contexts.
        session: SessionMemory instance.

    Returns:
        Dict with overall_f1, em_rate, answerable_f1, unanswerable_accuracy.
    """
    all_f1: list[float] = []
    all_em: list[int] = []
    answerable_f1: list[float] = []
    unanswerable_correct: list[int] = []
    unanswerable_total = 0

    for i, ex in enumerate(examples):
        try:
            result = answer_question(
                query=ex["question"],
                collection=collection,
                session=session,
            )
            pred = result["answer"]
            gold = ex["answer"]
            is_impossible = ex["is_impossible"]

            f1_val = f1_score(pred, gold)
            em_val = int(exact_match(pred, gold))

            all_f1.append(f1_val)
            all_em.append(em_val)

            if is_impossible:
                unanswerable_total += 1
                # Check if the model correctly said it can't answer
                no_answer_signals = [
                    "unanswerable", "no answer", "cannot be answered",
                    "not in the", "not mentioned", "does not contain",
                    "not provided", "i don't have",
                ]
                pred_lower = pred.lower()
                is_correct = any(s in pred_lower for s in no_answer_signals)
                unanswerable_correct.append(int(is_correct))
            else:
                answerable_f1.append(f1_val)

            if (i + 1) % 10 == 0:
                running_f1 = sum(all_f1) / len(all_f1)
                print(f"[eval] {i+1}/{len(examples)} — running F1: {running_f1:.4f}")

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"[eval] Error on example {i}: {e}")
            all_f1.append(0.0)
            all_em.append(0)
            time.sleep(2)  # back off on errors

    results = {
        "overall_f1": sum(all_f1) / len(all_f1) if all_f1 else 0.0,
        "em_rate": sum(all_em) / len(all_em) if all_em else 0.0,
        "answerable_f1": sum(answerable_f1) / len(answerable_f1) if answerable_f1 else 0.0,
        "unanswerable_accuracy": (
            sum(unanswerable_correct) / unanswerable_total
            if unanswerable_total > 0 else 0.0
        ),
        "n_evaluated": len(examples),
        "n_answerable": len(answerable_f1),
        "n_unanswerable": unanswerable_total,
    }

    return results


def print_summary(results: dict) -> None:
    """
    Print a formatted evaluation summary table.

    Args:
        results: Dict from run_evaluation with metric keys.
    """
    print("\n" + "=" * 55)
    print("  RAG Document Q&A — SQuAD 2.0 Evaluation Summary")
    print("=" * 55)
    print(f"  {'Metric':<30} {'Value':>10}")
    print("-" * 55)
    print(f"  {'Total Examples':<30} {results['n_evaluated']:>10}")
    print(f"  {'Answerable':<30} {results['n_answerable']:>10}")
    print(f"  {'Unanswerable':<30} {results['n_unanswerable']:>10}")
    print("-" * 55)
    print(f"  {'Overall F1':<30} {results['overall_f1']:>10.4f}")
    print(f"  {'Exact Match Rate':<30} {results['em_rate']:>10.4f}")
    print(f"  {'Answerable-only F1':<30} {results['answerable_f1']:>10.4f}")
    print(f"  {'Unanswerable Accuracy':<30} {results['unanswerable_accuracy']:>10.4f}")
    print("=" * 55)


def main():
    """Main evaluation entry point."""
    # Download SQuAD 2.0 dev set
    squad_data = download_squad_dev()

    # Extract 100 examples
    examples = extract_examples(squad_data, n=100)
    print(f"[eval] Extracted {len(examples)} examples "
          f"({sum(1 for e in examples if not e['is_impossible'])} answerable, "
          f"{sum(1 for e in examples if e['is_impossible'])} unanswerable)")

    # Set up a fresh ChromaDB for evaluation
    client = chromadb.Client()  # in-memory for evaluation
    eval_collection = client.get_or_create_collection("eval_chunks")

    # Ingest contexts
    print("[eval] Ingesting contexts into vector DB...")
    ingest_contexts(examples, eval_collection)

    # Run evaluation
    session = SessionMemory()
    print("[eval] Running evaluation...")
    results = run_evaluation(examples, eval_collection, session)

    # Print results
    print_summary(results)

    # Save results to file
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[eval] Results saved to evaluation_results.json")


if __name__ == "__main__":
    main()
