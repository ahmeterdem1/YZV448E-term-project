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

## How to run FastAPI server for PII cleaning

First start the Redis server with docker-compose:

```bash
docker-compose up -d
```

Then install the requirements:

```bash
pip install -r requirements.txt
```

Run the FastAPI server (this is run locally for now):

```bash
uvicorn app.main:app --reload
```

All endpoints are on `http://localhost:8000`. To send a text document, send a POST request to `/api/v1/process-text` 
with the following JSON body:

```json
{
  "text_content": "This is the text string I want to process with BERT."
}
```

The client receives an Item ID in response to this document. Then the client can send a GET request
to `/api/v1/text/{item_id}` to receive their document (if it is ready).

Sending a GET request to `/api/v1/queue` enables the user to see the pending documents.

There is a 30 second timeout on the queue. If 30 second elapses without processing, 
the queue is flushed for processing no matter how many items there are.