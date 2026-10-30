"""Run cross-paper retrieval, Gemini sufficiency checks, and grounded answering."""

from __future__ import annotations

import argparse

from classifier import SufficiencyClassifier
from generator import ABSTENTION, GeminiAnswerGenerator
from gemini_api import DEFAULT_GEMINI_MODEL
from retriever import DATA_PATH, SPLITS, HybridRetriever, print_results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--paper-id", help="Optional paper ID that the question concerns")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--initial-k", type=int, default=5)
    parser.add_argument("--round-k", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.initial_k < 1 or args.round_k < 1 or not 1 <= args.max_rounds <= 3:
        raise ValueError("initial-k and round-k must be positive; max-rounds must be 1-3")

    classifier = SufficiencyClassifier(args.gemini_model)
    retriever = HybridRetriever(DATA_PATH, split=args.split, device=args.device)
    contexts = []
    previous_incomplete = None

    for round_number in range(1, args.max_rounds + 1):
        k = min(
            len(retriever.paragraphs),
            args.initial_k + args.round_k * (round_number - 1),
        )
        search_query = args.query
        if round_number == 3 and previous_incomplete:
            terms = previous_incomplete.get("search_terms", [])
            missing = previous_incomplete.get("missing_evidence", "")
            expansion = list(terms)
            if missing and missing not in expansion:
                expansion.insert(0, missing)
            if expansion:
                search_query = (
                    f"{args.query}\n\nEvidence to look for: "
                    + "; ".join(expansion[:6])
                )
                print("Round 3 query expansion:", "; ".join(expansion[:6]))

        contexts = retriever.retrieve(search_query, k=k, alpha=args.alpha)
        cross_paper_count = len({item["paper_id"] for item in contexts})
        print(
            f"\nRetrieval round {round_number}: selected {len(contexts)} top paragraphs "
            f"across {cross_paper_count} papers (k={k})"
        )
        print_results(contexts)
        classification = classifier.predict(
            args.query, contexts, target_paper_id=args.paper_id
        )
        state = classification["label"]
        print(f"Classifier state: {state}")

        if state == "SUFFICIENT":
            generator = GeminiAnswerGenerator(
                args.gemini_model, client=classifier.client
            )
            answer = generator.generate(args.query, contexts, args.paper_id)
            print("\nAnswer:")
            print(answer)
            return
        if state == "NOT SUFFICIENT":
            print(f"\nAnswer:\n{ABSTENTION}")
            return
        if state != "INCOMPLETE":
            raise RuntimeError(f"Unexpected classifier state: {state!r}")
        previous_incomplete = classification
        if classification.get("missing_evidence"):
            print("Evidence gap:", classification["missing_evidence"])
        if classification.get("search_terms"):
            print("Suggested search terms:", "; ".join(classification["search_terms"]))

    print(f"\nAnswer:\n{ABSTENTION}")


if __name__ == "__main__":
    main()
