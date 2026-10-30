# CS561 Sufficiency-Aware QASPER RAG

A pipeline for preparing QASPER papers, retrieving top-ranked paragraphs across a complete split with a hybrid dense-and-sparse retriever, checking context sufficiency with Gemini, and generating a cited answer or abstention with Gemini.

The project converts QASPER's structured papers into one JSON file per paper and generates BGE paragraph embeddings. At runtime, it searches every paragraph in the selected split and combines normalized dense similarity with BM25. Gemini classifies the retrieved context as `SUFFICIENT`, `NOT SUFFICIENT`, or `INCOMPLETE`; incomplete context triggers two expanded retrieval rounds before the pipeline abstains.

## Repository structure

```text
.
├── main.py                         # Cross-paper retrieval + Gemini pipeline
├── retriever.py                    # Whole-split hybrid paragraph retriever
├── classifier.py                   # Gemini context-sufficiency classifier
├── generator.py                    # Gemini grounded answer generation
├── gemini_api.py                   # Shared Gemini API configuration
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

## Run retrieval, classification, and answer generation

Copy `.env.example` to `.env` and set your Gemini API key and preferred model:

```bash
cp .env.example .env
# Edit .env and replace the placeholder API key.
python main.py \
  --split test \
  --query "What is the seed lexicon?"
```

The application loads `GEMINI_API_KEY` and `GEMINI_MODEL` from the project-root
`.env` file. `.env` is ignored by Git; do not commit your API key. The
`--gemini-model` command-line option can override the model from `.env`.

The first round retrieves the top `--initial-k` paragraphs across the split (default 5). If Gemini labels the context `INCOMPLETE`, the pipeline increases `k` and retries the original query. For the third round, Gemini also identifies the missing evidence and suggests up to five short search phrases; the pipeline combines those phrases with the original question, increases `k` again, retrieves, and classifies once more. The search phrases are derived from the question and retrieved passages, and are treated only as retrieval hints. The pipeline stops after three classification rounds:

- `SUFFICIENT`: Gemini answers using only the retrieved context and cites paragraph IDs.
- `NOT SUFFICIENT`: immediately replies, “Not enough context to answer the question.”
- `INCOMPLETE`: retries retrieval twice; if still incomplete, replies, “Not enough context to answer the question.”

Set `--round-k` to control how many additional paragraphs each retry adds (default 5), or `--gemini-model` to choose another Gemini model. `--paper-id` is an optional subject-paper hint; retrieval remains cross-paper. Retrieval uses:

```text
alpha × min-max(dense cosine) + (1 - alpha) × min-max(BM25)
```

The default dense weight is `alpha = 0.7`. You can still search across a complete split with:

```bash
python retriever.py --split test --query "What is the seed lexicon?" --k 5
```

## Current verification

For QASPER paper `1909.00694` and the question “What is the seed lexicon?”, the annotated evidence paragraph is retrieved within the top five candidates. This validates the first-stage retrieval path; ranking quality must be evaluated over the complete QASPER test split rather than inferred from one example.

## Reproducibility

The QASPER Parquet revision and BGE model revision are pinned in `scripts/qasper.py`. The preparation script records its configuration and per-paper status under `data/manifests/` during each run.
