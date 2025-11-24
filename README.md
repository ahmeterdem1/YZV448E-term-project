# High Throughput PII Cleaner

This repo holds our YZV448E term project: building a **GDPR-friendly text cleaner** that can detect personally identifiable information (PII) in large batches of student essays and anonymize them before downstream use.

### What we built
- **Reusable data helpers** (`utils/data.py`): stratified train/val/test splitting plus convenient data loader builders.
- **Model wrapper** (`utils/model.py`): `HFPiiCleaner`, a Hugging Face token-classification pipeline that masks detected spans with configurable tokens (defaults to `[PII]`). Works with any BERT/DeBERTa-style NER checkpoint.
- **Inference + evaluation script** (`scripts/run_inference.py`): reads Kaggle’s JSON format, rebuilds text, runs the cleaner, and (for training data) reports character-level precision / recall / F1.
- **Requirements** (`requirements.txt`): everything you need to reproduce the pipeline, including optional serving dependencies (FastAPI + vLLM) for future API hosting.

### Data source
The system targets the [Kaggle “PII Detection / Removal from Educational Data” competition](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/overview), which provides ~22k anonymized essays with BIO token labels for seven PII categories (names, emails, usernames, IDs, phones, URLs, street addresses).

### Getting started
1. **Create & activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
   ```
2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Run inference/eval on a sample**
   ```bash
   python scripts/run_inference.py \
     --json-path data/train.json \
     --output-jsonl outputs/train_sample.jsonl \
     --limit 50 \
     --model-name dslim/bert-base-NER \
     --confidence-threshold 0.6
   ```
   For test-only inference, point `--json-path` to `data/test.json` (no metrics will be printed because labels are absent).

### Output & metrics
- The script writes one JSON line per essay: `{"document", "masked_text", "predicted_spans", "metrics?"}`.
- `masked_text` shows the essay with `[PII]` replacements.
- `predicted_spans` lists entity ranges and scores.
- `metrics` appears only when BIO labels exist (train data) and contains per-document character-level precision, recall, and F1, matching the competition scoring style.

### Next steps
- Experiment with different checkpoints (e.g., DeBERTa-v3-large) using the same wrapper.
- Plug the cleaner into a FastAPI service for batch processing.
- Explore vLLM / Triton serving if we move to larger generative models for rule-free masking.


