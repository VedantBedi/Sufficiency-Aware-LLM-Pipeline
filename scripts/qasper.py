#!/usr/bin/env python3
"""Create one paragraph-and-embedding JSON file for every QASPER paper.

Output: data/papers_json/<split>/<paper_id>.json
    {"paper_id": ..., "paragraphs": [{"para_id", "section", "content", "dense_embedding"}, ...]}

Edit the CONFIGURATION section and run directly (CLI flags are optional overrides).
"""

import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "papers_json"
CACHE_ROOT = PROJECT_ROOT / "data" / "cache"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "qasper_build_manifest.jsonl"
SUMMARY_PATH = PROJECT_ROOT / "data" / "manifests" / "qasper_build_summary.json"

DATASET_NAME = "allenai/qasper"
DATASET_CONFIG = "qasper"
# Pinned Parquet conversion (newer `datasets` versions can't run the old qasper.py script).
DATASET_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
SPLITS = ("train", "validation", "test")
PARQUET_FILES = {
    split: f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/"
           f"{DATASET_REVISION}/{DATASET_CONFIG}/{split}/0000.parquet"
    for split in SPLITS
}

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BATCH_SIZE = 8
DEVICE = "auto"  # "auto", "cuda", "mps", or "cpu"
NORMALIZE_EMBEDDINGS = True
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_DECIMALS = 6

INCLUDE_ABSTRACT = True
FORCE_REBUILD = False
LIMIT_PAPERS_PER_SPLIT = None  # e.g. 2 for a quick pilot; None = all papers
# =============================================================================

# Must be set before importing datasets / sentence_transformers.
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "datasets"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")  # stop transformers from importing TensorFlow if installed

# Import order matters on macOS: pyarrow (datasets) must load before torch.
from datasets import load_dataset  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
import torch  # noqa: E402

# Settings that affect the output; hashed into the summary file.
BUILD_CONFIG = {
    "dataset_name": DATASET_NAME,
    "dataset_revision": DATASET_REVISION,
    "embedding_model": EMBEDDING_MODEL,
    "embedding_model_revision": EMBEDDING_MODEL_REVISION,
    "normalize_embeddings": NORMALIZE_EMBEDDINGS,
    "max_sequence_length": MAX_SEQUENCE_LENGTH,
    "include_abstract": INCLUDE_ABSTRACT,
    "embedding_decimals": EMBEDDING_DECIMALS,
}


def choose_device(requested):
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        return next(name for name, ok in available.items() if ok)
    if requested not in available:
        raise ValueError(f"Unsupported device: {requested}")
    if not available[requested]:
        raise RuntimeError(f"DEVICE={requested!r} was requested but is unavailable.")
    return requested


def clean_text(value):
    """Collapse whitespace (incl. non-breaking spaces); None -> ''."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u00a0", " ")).strip()


def safe_filename(paper_id):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id.strip())
    if not safe or safe in {".", ".."}:
        safe = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:16]
    return safe


def extract_paragraphs(record, start_id):
    """Flatten abstract + full text into [{para_id, section, content}], IDs from start_id."""
    sections = []  # (section, content)

    abstract = clean_text(record.get("abstract"))
    if INCLUDE_ABSTRACT and abstract:
        sections.append(("Abstract", abstract))

    full_text = record.get("full_text") or {}
    names = list(full_text.get("section_name") or [])
    texts = list(full_text.get("paragraphs") or [])
    if len(names) != len(texts):
        raise ValueError(
            f"section_name has {len(names)} entries but paragraphs has {len(texts)} entries"
        )

    for i, (name, section_paragraphs) in enumerate(zip(names, texts), start=1):
        section = clean_text(name) or f"Untitled section {i}"
        if isinstance(section_paragraphs, str):
            section_paragraphs = [section_paragraphs]
        for raw in section_paragraphs or []:
            content = clean_text(raw)
            if content:
                sections.append((section, content))

    if not sections:
        raise ValueError("paper has no non-empty abstract or full-text paragraphs")

    return [
        {"para_id": start_id + i, "section": section, "content": content}
        for i, (section, content) in enumerate(sections)
    ]


def free_gpu_memory():
    """Release cached accelerator memory (no-op on CPU)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def embed(model, paragraphs, batch_size):
    """Add `dense_embedding` to each paragraph; returns the batch size that worked.

    On an out-of-memory error: free cached memory and halve the batch size. If it
    still fails at batch size 1, move the model to the CPU for the rest of the run.
    """
    contents = [p["content"] for p in paragraphs]
    while True:
        try:
            vectors = model.encode(
                contents,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=NORMALIZE_EMBEDDINGS,
            )
            break
        except RuntimeError as exc:  # CUDA/MPS OOM errors are RuntimeErrors
            if "out of memory" not in str(exc).lower():
                raise
            free_gpu_memory()
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                print(f"\nOut of memory; retrying with batch size {batch_size}")
            elif model.device.type != "cpu":
                print("\nOut of memory at batch size 1; switching the model to CPU")
                model.to("cpu")
                batch_size = BATCH_SIZE
            else:
                raise
    if len(vectors) != len(paragraphs):
        raise ValueError(f"Model returned {len(vectors)} embeddings for {len(paragraphs)} paragraphs")
    for p, vec in zip(paragraphs, vectors):
        p["dense_embedding"] = [round(float(x), EMBEDDING_DECIMALS) for x in vec]
    return batch_size


def write_json(path, payload):
    """Write compact JSON atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    tmp.replace(path)


def already_built(path, paper_id, start_id, count):
    """True if `path` holds a finished file with the expected paper, IDs and embeddings."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        paragraphs = data.get("paragraphs") or []
        return (
            data.get("paper_id") == paper_id
            and len(paragraphs) == count
            and all(
                p.get("para_id") == start_id + i
                and p.get("content")
                and isinstance(p.get("dense_embedding"), list)
                and p["dense_embedding"]
                for i, p in enumerate(paragraphs)
            )
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild existing JSON files")
    parser.add_argument("--limit", type=int, default=LIMIT_PAPERS_PER_SPLIT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer")
    force = FORCE_REBUILD or args.force

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Loading structured Parquet files for {DATASET_NAME}...")
    dataset = load_dataset(
        "parquet", data_files=PARQUET_FILES, cache_dir=str(CACHE_ROOT / "datasets")
    )

    print(f"Loading {EMBEDDING_MODEL} on {device}...")
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
        cache_folder=str(CACHE_ROOT / "models"),
        revision=EMBEDDING_MODEL_REVISION,
    )
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    dim = model.get_sentence_embedding_dimension()

    batch_size = args.batch_size  # may shrink if the device runs out of memory
    counts = {"written": 0, "skipped": 0, "failed": 0, "paragraphs": 0}
    manifest = []
    next_id = 1  # corpus-wide paragraph counter, shared across all splits

    for split in SPLITS:
        if split not in dataset:
            raise KeyError(f"Dataset does not contain expected split: {split}")
        records = list(dataset[split])[: args.limit]
        split_dir = OUTPUT_ROOT / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for record in tqdm(records, desc=f"Building {split}", unit="paper"):
            paper_id = clean_text(record.get("id"))
            if not paper_id:
                counts["failed"] += 1
                manifest.append({"split": split, "status": "failed", "error": "missing paper id"})
                continue

            try:
                start_id = next_id
                paragraphs = extract_paragraphs(record, start_id)
                count = len(paragraphs)
                # Advance before embedding so later IDs stay deterministic even if
                # this paper fails below.
                next_id += count
                counts["paragraphs"] += count

                path = split_dir / f"{safe_filename(paper_id)}.json"
                status = "skipped"
                if force or not already_built(path, paper_id, start_id, count):
                    batch_size = embed(model, paragraphs, batch_size)
                    free_gpu_memory()
                    write_json(path, {"paper_id": paper_id, "paragraphs": paragraphs})
                    status = "written"
                counts[status] += 1

                row = {
                    "paper_id": paper_id,
                    "split": split,
                    "status": status,
                    "paragraph_count": count,
                    "paragraph_id_start": start_id,
                    "paragraph_id_end": start_id + count - 1,
                }
                if status == "written":
                    row["embedding_dimension"] = dim
                row["path"] = str(path)
                manifest.append(row)
            except Exception as exc:  # one bad paper shouldn't kill a long run
                counts["failed"] += 1
                manifest.append(
                    {"paper_id": paper_id, "split": split, "status": "failed", "error": repr(exc)}
                )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fingerprint = hashlib.sha256(
        json.dumps(BUILD_CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary = {
        "config": BUILD_CONFIG,
        "config_fingerprint": fingerprint,
        "device": model.device.type,  # final device (may differ from `device` after an OOM fallback)
        "embedding_dimension": dim,
        "counts": counts,
        "manifest": str(MANIFEST_PATH),
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())