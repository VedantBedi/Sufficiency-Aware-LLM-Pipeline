# CS561 Sufficiency-Aware QASPER RAG

A local pipeline for preparing QASPER papers, retrieving paper-specific evidence with a hybrid dense-and-sparse retriever, checking sufficiency, and generating a cited answer or abstention.

The project converts QASPER's structured papers into one JSON file per paper, generates BGE paragraph embeddings, and retrieves candidate evidence using normalized dense similarity combined with BM25. It builds round-wise classifier labels from QASPER's own evidence annotations and fine-tunes Qwen2.5-0.5B-Instruct with LoRA. It is designed to run locally, including on Apple Silicon.

## Repository structure

```text
.
├── main.py                         # Retrieval + sufficiency-check loop
├── retriever.py                    # Lazy per-paper hybrid retriever
├── classifier.py                   # LoRA classifier inference
├── generator.py                    # Small LLM answer/abstention generation
├── requirements.txt                # Python dependencies
└── scripts/
    ├── qasper.py                   # QASPER download, parsing, and embedding
    ├── build_classifier_data.py    # Retrieval states labeled from QASPER evidence
    └── train_classifier.py         # Qwen2.5-0.5B LoRA training
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

## Build sufficiency classifier data

Prepare the paper JSON/embeddings first, then build separate train and validation examples:

```bash
python scripts/build_classifier_data.py --splits train validation --initial-k 5 --cross-paper-k 5 --rounds 3
```

The script reads the pinned QASPER Parquet source and aligns each annotator's evidence text to the prepared paragraph IDs. Round one retrieves only within that question's paper. If evidence is still missing, later example states add fresh passages from whole-split retrieval, which can include other papers. It emits these states into `data/classifier/{train,validation}.jsonl`. Labels mean:

- `SUFFICIENT`: all evidence for this annotation is present in the current passages.
- `DISTRIBUTED`: the question is answerable, but this retrieval state is missing annotated evidence and needs another round.
- `UNANSWERABLE`: the QASPER answer annotation says the paper does not answer the question.

Thus `DISTRIBUTED` is the operational “retrieve more” state; it can include a localized answer missed by the current retriever as well as genuinely distributed evidence. Cross-paper passages are treated as additional context, not as proof of what the target paper states. Examples are created per annotation to avoid combining different annotators' evidence into artificial gold sets. Exact normalized evidence-to-paragraph matches are required; unmatched evidence is not silently guessed.

## Train the classifier

Install the full requirements and train against the original QASPER train/validation splits:

```bash
python scripts/train_classifier.py --epochs 3 --max-length 4096
```

The model is saved under `models/sufficiency-qwen-lora/`; generated model files are excluded from Git. A small pilot can use `--max-questions` when building the data, then pass the generated JSONL paths explicitly to the trainer. Do not use QASPER test data for training or model selection.

## Run retrieval, classification, and answer generation

Run the full pipeline for one target paper:

```bash
python main.py \
  --split test \
  --paper-id 1909.00694 \
  --query "What is the seed lexicon?" \
  --adapter models/sufficiency-qwen-lora
```

The pipeline runs at most three rounds. Round one retrieves five passages from the target paper. If the classifier returns `DISTRIBUTED`, the next rounds add five new passages per round from retrieval over the full split, including other papers. The classifier then passes the final state and accumulated passages to the small Qwen answer generator:

- `SUFFICIENT`: answer from supported evidence and cite passage IDs.
- `DISTRIBUTED`: provide a cautious, evidence-based answer with the warning “Not enough context; please cross-check this answer.”
- `UNANSWERABLE`: abstain with “No context in the retrieved material; I can’t answer this from the paper.”

For `DISTRIBUTED`, the extra papers may help answer a broader question, but the system labels citations with their paper IDs and warns the user to cross-check. Use a split containing the selected target paper. Retrieval uses the same hybrid score as the cross-paper CLI:

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
