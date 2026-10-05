"""Hybrid paragraph retriever over a whole QASPER split (train / validation / test).

Every paragraph of every paper in the chosen split is searched at once, scored
with normalized BGE cosine similarity combined with BM25. Expects the output of
build_qasper_json.py: data/papers_json/<split>/<paper_id>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

# Must be set before sentence_transformers/transformers are imported: stops
# transformers from loading TensorFlow (a common cause of the macOS
# "mutex.cc : 452 RAW: Lock blocking" hang) even if it is installed.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "papers_json"
SPLITS = ("train", "validation", "test")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL_CACHE = PROJECT_ROOT / "data" / "cache" / "models"
EMBED_DIM = 384
MAX_SEQUENCE_LENGTH = 512
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norm, 1e-12, None)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _minmax(array: np.ndarray) -> np.ndarray:
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-12:
        return np.zeros_like(array)
    return (array - low) / (high - low)


class HybridRetriever:
    """Retrieve the best paragraphs from every paper in one QASPER split."""

    def __init__(
        self,
        data_path: str | Path = DATA_PATH,
        split: str | None = "train",
        model_name: str = MODEL_NAME,
        device: str | None = None,
    ) -> None:
        """split: "train", "validation" or "test"; None searches all three."""
        if split is not None and split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS} or None, got {split!r}")
        self.data_path = Path(data_path).expanduser().resolve()
        self.split = split
        self.paragraphs: list[dict[str, Any]] = []
        self._load_corpus()

        kwargs: dict[str, Any] = {"cache_folder": str(MODEL_CACHE)}
        if model_name == MODEL_NAME:
            kwargs["revision"] = MODEL_REVISION
        if device is not None:
            kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **kwargs)
        self.model.max_seq_length = MAX_SEQUENCE_LENGTH

        get_dim = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension"
        )
        dimension = get_dim()
        if dimension != EMBED_DIM:
            raise ValueError(
                f"Query model produces {dimension}-dimensional vectors; stored "
                f"paragraph vectors require {EMBED_DIM}."
            )

    def _load_corpus(self) -> None:
        if self.split is None:
            root, pattern = self.data_path, "*/*.json"
        else:
            root, pattern = self.data_path / self.split, "*.json"
        paths = sorted(root.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No paper JSON files found in {root}")

        blocks: list[np.ndarray] = []
        tokenized: list[list[str]] = []
        for path in tqdm(paths, desc=f"Loading {self.split or 'all'}", unit="paper"):
            paper = json.loads(path.read_text(encoding="utf-8"))
            paper_id = str(paper.get("paper_id", "")).strip()
            paragraphs = paper.get("paragraphs")
            if not paper_id or not isinstance(paragraphs, list) or not paragraphs:
                raise ValueError(f"{path}: missing paper_id or paragraphs")
            try:
                embeddings = np.asarray(
                    [p["dense_embedding"] for p in paragraphs], dtype=np.float32
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: malformed embeddings") from exc
            if embeddings.shape != (len(paragraphs), EMBED_DIM):
                raise ValueError(
                    f"{path}: embedding matrix has shape {embeddings.shape}; "
                    f"expected ({len(paragraphs)}, {EMBED_DIM})."
                )
            blocks.append(embeddings)
            for p in paragraphs:
                self.paragraphs.append(
                    {
                        "paper_id": paper_id,
                        "para_id": p["para_id"],
                        "section": p.get("section", ""),
                        "content": p["content"],
                    }
                )
                tokenized.append(_tokenize(p["content"]))

        self.embeddings = _l2_normalize(np.vstack(blocks))
        self.bm25 = BM25Okapi(tokenized)
        print(f"Loaded {len(paths)} papers / {len(self.paragraphs)} paragraphs from {root}")

    def _encode_query(self, query: str) -> np.ndarray:
        vector = self.model.encode(
            QUERY_PREFIX + query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)

    def retrieve(self, query: str, k: int = 5, alpha: float = 0.7) -> list[dict[str, Any]]:
        """Return the top-k paragraphs across the whole split, best first."""
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        tokens = _tokenize(query)
        if not tokens:
            raise ValueError("query must contain at least one letter or number")

        dense = self.embeddings @ self._encode_query(query)
        bm25 = np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)
        scores = alpha * _minmax(dense) + (1.0 - alpha) * _minmax(bm25)

        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]

        return [
            {
                "rank": rank,
                **self.paragraphs[int(i)],
                "score": float(scores[i]),
                "dense_score": float(dense[i]),
                "bm25_score": float(bm25[i]),
            }
            for rank, i in enumerate(top, start=1)
        ]


def print_results(results: list[dict[str, Any]], preview_characters: int = 300) -> None:
    for r in results:
        content = r["content"].replace("\n", " ")
        suffix = "..." if len(content) > preview_characters else ""
        print(
            f"#{r['rank']}  paper={r['paper_id']}  para_id={r['para_id']}  "
            f"section={r['section']!r}  score={r['score']:.4f} "
            f"(dense={r['dense_score']:.3f}, bm25={r['bm25_score']:.2f})"
        )
        print(f"    {content[:preview_characters]}{suffix}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    args = parser.parse_args()

    retriever = HybridRetriever(args.data, split=args.split, device=args.device)
    print_results(retriever.retrieve(args.query, k=args.k, alpha=args.alpha))


if __name__ == "__main__":
    main()