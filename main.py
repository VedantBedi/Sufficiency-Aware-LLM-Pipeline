"""Run paper-specific retrieval, cross-paper recovery, classification, and answering."""

from __future__ import annotations

import argparse

from classifier import DEFAULT_ADAPTER, SufficiencyClassifier
from generator import DEFAULT_GENERATOR_MODEL, ABSTENTION, SmallAnswerGenerator
from retriever import DATA_PATH, SPLITS, HybridRetriever, print_results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Question about the selected paper")
    parser.add_argument("--paper-id", required=True, help="QASPER/arXiv paper ID")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--initial-k", type=int, default=5)
    parser.add_argument("--round-k", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.initial_k < 1 or args.round_k < 1 or not 1 <= args.max_rounds <= 3:
        raise ValueError("initial-k and round-k must be positive; max-rounds must be 1-3")
    retriever = HybridRetriever(DATA_PATH, split=args.split, device=args.device)
    classifier = SufficiencyClassifier(args.adapter, device=args.device)

    contexts = retriever.retrieve(
        args.query, k=args.initial_k, alpha=args.alpha, paper_id=args.paper_id
    )
    seen = {(p["paper_id"], int(p["para_id"])) for p in contexts}
    print(f"\nRetrieval round 1: {len(contexts)} passages from target paper")
    print_results(contexts)
    state = classifier.predict(args.query, contexts, target_paper_id=args.paper_id)
    print(f"Classifier state: {state['label']}")

    for round_number in range(2, args.max_rounds + 1):
        if state["label"] != "DISTRIBUTED":
            break
        # DISTRIBUTED triggers expanded search over the full split. Candidate
        # paragraphs from the target and other papers are both eligible.
        candidate_k = min(
            len(retriever.paragraphs),
            max(200, args.initial_k * 3 + args.round_k * (args.max_rounds - 1)),
        )
        global_results = retriever.retrieve(
            args.query, k=candidate_k, alpha=args.alpha
        )
        additions = []
        other_papers = [p for p in global_results if p["paper_id"] != args.paper_id]
        target_paper_more = [p for p in global_results if p["paper_id"] == args.paper_id]
        for passage in other_papers + target_paper_more:
            key = (passage["paper_id"], int(passage["para_id"]))
            if key not in seen:
                seen.add(key)
                additions.append(passage)
                if len(additions) >= args.round_k:
                    break
        if not additions:
            print("No new passages found in the expanded search.")
            break
        contexts.extend(additions)
        external_count = sum(p["paper_id"] != args.paper_id for p in additions)
        print(
            f"\nRetrieval round {round_number}: added {len(additions)} expanded-split passages "
            f"({external_count} from other papers)"
        )
        print_results(additions)
        state = classifier.predict(args.query, contexts, target_paper_id=args.paper_id)
        print(f"Classifier state: {state['label']}")

    print("\nAnswer:")
    if state["label"] == "UNANSWERABLE":
        print(ABSTENTION)
        return
    generator = SmallAnswerGenerator(args.generator_model, device=args.device)
    answer = generator.generate(
        args.query, contexts, state["label"], target_paper_id=args.paper_id
    )
    print(answer)


if __name__ == "__main__":
    main()
