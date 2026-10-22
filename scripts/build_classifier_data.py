#!/usr/bin/env python3
"""Build paper-grounded, round-wise sufficiency classifier examples from QASPER.

Labels describe the current retrieval state: SUFFICIENT means all annotated
evidence for this annotation is present; DISTRIBUTED means the question is
answerable but the current context still misses evidence (retrieve again);
UNANSWERABLE follows the QASPER annotation. After the first paper-restricted
round, context grows with globally retrieved paragraphs from the full split,
including other papers. QASPER evidence is aligned to paragraph IDs using
normalized exact text matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retriever import DATA_PATH, SPLITS, HybridRetriever  # noqa: E402

DATASET_NAME = "allenai/qasper"
DATASET_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
DATASET_CONFIG = "qasper"
PARQUET_FILES = {
    split: f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/"
    f"{DATASET_REVISION}/{DATASET_CONFIG}/{split}/0000.parquet"
    for split in SPLITS
}
OUTPUT_DIR = ROOT / "data" / "classifier"


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\u00a0", " ")).strip().casefold()


def safe_filename(paper_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id.strip())
    if not safe or safe in {".", ".."}:
        safe = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:16]
    return safe


def question_annotations(qas: dict[str, Any], index: int) -> list[dict[str, Any]]:
    """Normalize the nested QASPER feature into a list of annotator answers."""
    all_answers = qas.get("answers") or []
    item = all_answers[index] if index < len(all_answers) else None
    if isinstance(item, list):
        return [a for a in item if isinstance(a, dict)]
    if not isinstance(item, dict):
        return []
    # Parquet's nested feature is a struct of parallel arrays.
    answers = item.get("answer")
    if isinstance(answers, list):
        return [a for a in answers if isinstance(a, dict)]
    return [item]


def gold_ids(annotation: dict[str, Any], content_to_ids: dict[str, list[int]]) -> list[int]:
    evidence = annotation.get("evidence") or []
    found: set[int] = set()
    for text in evidence:
        key = normalize(text)
        found.update(content_to_ids.get(key, []))
    return sorted(found)


def build_split(split: str, args: argparse.Namespace) -> int:
    split_dir = DATA_PATH / split
    if not split_dir.exists() or not any(split_dir.glob("*.json")):
        raise FileNotFoundError(f"No prepared paper JSON for split {split!r}: {split_dir}")
    print(f"Loading QASPER {split} and preparing retrieval...")
    dataset = load_dataset(
        "parquet", data_files={split: PARQUET_FILES[split]}, split=split,
        cache_dir=str(ROOT / "data" / "cache" / "datasets"),
    )
    retriever = HybridRetriever(DATA_PATH, split=split, device=args.device)
    output = OUTPUT_DIR / f"{split}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    with output.open("w", encoding="utf-8") as f:
        for record in tqdm(dataset, desc=f"Building {split} examples", unit="paper"):
            paper_id = str(record.get("id", "")).strip()
            if not paper_id:
                skipped += 1
                continue
            qas = record.get("qas") or {}
            questions = qas.get("question") or []
            try:
                paper = json.loads(
                    (split_dir / f"{safe_filename(paper_id)}.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                skipped += len(questions)
                continue
            if paper.get("paper_id") != paper_id:
                skipped += len(questions)
                continue
            paragraphs = paper.get("paragraphs") or []
            content_to_ids: dict[str, list[int]] = {}
            for p in paragraphs:
                content_to_ids.setdefault(normalize(p.get("content")), []).append(int(p["para_id"]))

            for qi, raw_question in enumerate(questions):
                question = str(raw_question or "").strip()
                if not question:
                    skipped += 1
                    continue
                annotations = question_annotations(qas, qi)
                if not annotations:
                    skipped += 1
                    continue
                try:
                    ranked = retriever.retrieve(
                        question, k=args.initial_k, alpha=args.alpha, paper_id=paper_id
                    )
                except (KeyError, ValueError):
                    skipped += len(annotations)
                    continue
                if not ranked:
                    skipped += len(annotations)
                    continue

                # Mimic runtime: round one searches the target paper. Each later
                # round adds fresh results from retrieval over the whole split.
                candidate_k = min(
                    len(retriever.paragraphs),
                    max(200, args.initial_k * 3 + args.cross_paper_k * (args.rounds - 1)),
                )
                global_ranked = retriever.retrieve(
                    question, k=candidate_k, alpha=args.alpha
                )
                stages = [ranked]
                seen = {(p["paper_id"], int(p["para_id"])) for p in ranked}
                accumulated = list(ranked)
                for _ in range(1, args.rounds):
                    # Prefer passages from different papers so distributed
                    # examples genuinely teach cross-paper expansion. If the
                    # global top candidates do not provide enough, fill the
                    # remainder with fresh target-paper passages.
                    fresh = []
                    ordered_candidates = [
                        p for p in global_ranked if p["paper_id"] != paper_id
                    ] + [p for p in global_ranked if p["paper_id"] == paper_id]
                    for passage in ordered_candidates:
                        key = (passage["paper_id"], int(passage["para_id"]))
                        if key not in seen:
                            seen.add(key)
                            fresh.append(passage)
                            if len(fresh) >= args.cross_paper_k:
                                break
                    if not fresh:
                        break
                    accumulated.extend(fresh)
                    stages.append(list(accumulated))

                # Keep each annotator's evidence as its own supervision signal;
                # this avoids inventing a larger gold set by unioning workers.
                for ai, annotation in enumerate(annotations):
                    is_unanswerable = bool(annotation.get("unanswerable", False))
                    evidence_ids = gold_ids(annotation, content_to_ids)
                    if not is_unanswerable and not evidence_ids:
                        skipped += 1  # no usable evidence supervision
                        continue
                    for round_number, context in enumerate(stages, start=1):
                        observed_target_evidence = {
                            int(p["para_id"]) for p in context
                            if p["paper_id"] == paper_id
                        }
                        complete = set(evidence_ids).issubset(observed_target_evidence)
                        label = (
                            "UNANSWERABLE" if is_unanswerable
                            else "SUFFICIENT" if complete
                            else "DISTRIBUTED"
                        )
                        row = {
                            "split": split,
                            "paper_id": paper_id,
                            "question_id": (qas.get("question_id") or [None] * len(questions))[qi],
                            "annotation_index": ai,
                            "target_paper_id": paper_id,
                            "question": question,
                            "contexts": [
                                {"paper_id": p["paper_id"], "para_id": int(p["para_id"]),
                                 "section": p.get("section", ""),
                                 "content": p["content"], "score": float(p["score"])}
                                for p in context
                            ],
                            "label": label,
                            "gold_evidence_para_ids": evidence_ids,
                            "retrieval_round": round_number,
                            "context_count": len(context),
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
                        if args.max_questions and written >= args.max_questions:
                            break
                        # Runtime stops retrieval when classifier should answer
                        # or abstain; do not train unreachable later states.
                        if label != "DISTRIBUTED":
                            break
                    if args.max_questions and written >= args.max_questions:
                        break
                if args.max_questions and written >= args.max_questions:
                    break
    print(f"Wrote {written} examples to {output}; skipped {skipped} annotations/questions.")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["train", "validation"])
    parser.add_argument("--initial-k", type=int, default=5)
    parser.add_argument("--cross-paper-k", type=int, default=5,
                        help="Add this many new globally retrieved passages per later round.")
    parser.add_argument("--rounds", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Stop after this many emitted examples per split (useful for pilots).")
    args = parser.parse_args()
    if args.initial_k < 1 or args.cross_paper_k < 1 or args.rounds < 1:
        parser.error("initial-k, cross-paper-k and rounds must all be positive")
    if not 0 <= args.alpha <= 1:
        parser.error("alpha must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    for split in args.splits:
        build_split(split, args)


if __name__ == "__main__":
    main()
