"""Run one QASPER hybrid-retrieval example directly from VS Code."""

from retriever import DATA_PATH, HybridRetriever, print_results

SPLIT = "train"  # "train", "validation", "test", or None for all three
QUESTION = "What is Clinical text structuring ?"
TOP_K = 5
DENSE_WEIGHT = 0.7
DEVICE = None  # None selects automatically; use "cuda", "mps", or "cpu" to override.


def main() -> None:
    retriever = HybridRetriever(DATA_PATH, split=SPLIT, device=DEVICE)
    results = retriever.retrieve(QUESTION, k=TOP_K, alpha=DENSE_WEIGHT)
    print_results(results)


if __name__ == "__main__":
    main()