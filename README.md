# CS561 QASPER Evidence Retrieval

A local, reproducible pipeline for preparing QASPER papers and retrieving evidence paragraphs with a hybrid dense-and-sparse retriever.

The project converts QASPER's structured papers into one JSON file per paper, generates BGE paragraph embeddings, and retrieves candidate evidence using normalized dense similarity combined with BM25. It is designed to run locally on an Apple Silicon Mac.

## Repository structure

```text
.
├── main.py                         # Editable retrieval example for VS Code
├── retriever.py                    # Lazy per-paper hybrid retriever
├── requirements.txt                # Python dependencies
└── scripts/
    └── qasper.py                   # QASPER download, parsing, and embedding
```

Generated data, downloaded models, virtual environments, and logs are excluded from Git.

## Setup

Python 3.9 or newer is supported. The pipeline runs on Windows, macOS, and Linux
with CPU inference. It automatically uses CUDA on supported NVIDIA systems or
MPS on supported Apple Silicon systems; accelerator support depends on the
installed PyTorch build and drivers.

```bash
python3 -m pip install -r requirements.txt
```

## Prepare QASPER

Open `scripts/qasper.py` and edit the configuration block if needed. For a quick pilot, set:

```python
LIMIT_PAPERS_PER_SPLIT = 2
```

Use `None` to process all 1,585 papers. Run:

```bash
python scripts/qasper.py
```

The script reads Hugging Face's maintained Parquet conversion, preserves the official train/validation/test split, and writes one JSON per paper under `data/papers_json/`. Paragraph IDs increase globally through train, validation, and test. Interrupted runs are resumable.

The pinned embedding model is `BAAI/bge-small-en-v1.5`, producing normalized 384-dimensional vectors. The first run requires an internet connection to download the dataset and model. Their cache and generated data are stored under this project.

## Run retrieval

Edit the variables at the top of `main.py` and run it directly from VS Code, or use the command line:

```bash
python retriever.py \
  --split test \
  --query "What is the seed lexicon?" \
  --k 5
```

The retriever loads all generated papers in the selected split and searches their paragraphs together. Use `--split train`, `--split validation`, or `--split test`. Its final score is:

```text
alpha × min-max(dense cosine) + (1 - alpha) × min-max(BM25)
```

The default dense weight is `alpha = 0.7`.

## Current verification

For QASPER paper `1909.00694` and the question “What is the seed lexicon?”, the annotated evidence paragraph is retrieved within the top five candidates. This validates the first-stage retrieval path; ranking quality must be evaluated over the complete QASPER test split rather than inferred from one example.

## Reproducibility

The QASPER Parquet revision and BGE model revision are pinned in `scripts/qasper.py`. The preparation script records its configuration and per-paper status under `data/manifests/` during each run.
