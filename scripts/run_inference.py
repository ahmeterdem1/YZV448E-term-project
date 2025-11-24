"""
Inference/evaluation helper tailored to the Kaggle PII dataset (JSON format).

* Accepts either train or test JSON files published by Kaggle.
* Automatically rebuilds the original text from tokens + whitespace when needed.
* Converts BIO token labels into character-level spans for scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Ensure project root is on PYTHONPATH when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from utils.model import EntitySpan, HFPiiCleaner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PII inference on Kaggle JSON.")
    parser.add_argument(
        "--json-path",
        type=Path,
        required=True,
        help="Path to Kaggle {train|test}.json.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Where to write predictions/masked text (JSONL).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dslim/bert-base-NER",
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of documents to process (0 = all).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum confidence required to keep a predicted span.",
    )
    parser.add_argument(
        "--aggregation-strategy",
        type=str,
        default="simple",
        choices=["none", "simple", "first", "average", "max"],
        help="Hugging Face aggregation strategy.",
    )
    parser.add_argument(
        "--mask-token",
        type=str,
        default="[PII]",
        help="Replacement string for detected spans.",
    )
    return parser.parse_args()


def rebuild_text_and_offsets(
    tokens: Sequence[str], trailing_whitespace: Sequence[bool]
) -> Tuple[str, List[Tuple[int, int]]]:
    if len(tokens) != len(trailing_whitespace):
        raise ValueError("tokens and trailing_whitespace lengths differ.")

    parts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0

    for token, has_space in zip(tokens, trailing_whitespace):
        parts.append(token)
        start = cursor
        cursor += len(token)
        offsets.append((start, cursor))
        if has_space:
            parts.append(" ")
            cursor += 1

    full_text = "".join(parts)
    return full_text, offsets


def bio_to_spans(
    labels: Sequence[str], offsets: Sequence[Tuple[int, int]]
) -> List[Dict[str, int]]:
    spans: List[Dict[str, int]] = []
    current_label: Optional[str] = None
    current_start: Optional[int] = None
    prev_end: Optional[int] = None

    for label, (start, end) in zip(labels, offsets):
        if label == "O":
            if current_label is not None:
                spans.append(
                    {
                        "label": current_label,
                        "start": current_start,  # type: ignore[arg-type]
                        "end": prev_end,  # type: ignore[arg-type]
                    }
                )
                current_label = None
                current_start = None
            prev_end = end
            continue

        tag, _, pii_type = label.partition("-")
        if tag == "B" or (tag == "I" and current_label != pii_type):
            if current_label is not None:
                spans.append(
                    {
                        "label": current_label,
                        "start": current_start,  # type: ignore[arg-type]
                        "end": prev_end,  # type: ignore[arg-type]
                    }
                )
            current_label = pii_type
            current_start = start
        prev_end = end

    if current_label is not None:
        spans.append(
            {
                "label": current_label,
                "start": current_start,  # type: ignore[arg-type]
                "end": prev_end,  # type: ignore[arg-type]
            }
        )
    return spans


def spans_to_index_set(spans: Sequence[Dict[str, int]]) -> Set[int]:
    idxs: Set[int] = set()
    for span in spans:
        idxs.update(range(int(span["start"]), int(span["end"])))
    return idxs


def entity_spans_to_dict(spans: Sequence[EntitySpan]) -> List[Dict[str, object]]:
    return [
        {"start": span.start, "end": span.end, "label": span.label, "score": span.score}
        for span in spans
    ]


def precision_recall_f1(pred: Set[int], truth: Set[int]) -> Dict[str, float]:
    if not pred and not truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    args = parse_args()

    essays = json.loads(args.json_path.read_text(encoding="utf-8"))
    if args.limit > 0:
        essays = essays[: args.limit]
    if not essays:
        raise SystemExit("No essays loaded. Check json-path/limit.")

    cleaner = HFPiiCleaner(
        model_name_or_path=args.model_name,
        aggregation_strategy=args.aggregation_strategy,
        confidence_threshold=args.confidence_threshold,
        mask_token=args.mask_token,
    )

    metrics: List[Dict[str, float]] = []
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.output_jsonl.open("w", encoding="utf-8") as writer:
        for essay in essays:
            doc_id = str(essay.get("document", essay.get("id", "unknown")))
            tokens = essay["tokens"]
            trailing = essay["trailing_whitespace"]
            full_text = essay.get("full_text")

            if not full_text:
                full_text, offsets = rebuild_text_and_offsets(tokens, trailing)
            else:
                _, offsets = rebuild_text_and_offsets(tokens, trailing)

            pred_spans = cleaner.forward([full_text])[0]
            masked_text = cleaner.mask_text(full_text)

            payload = {
                "document": doc_id,
                "masked_text": masked_text,
                "predicted_spans": entity_spans_to_dict(pred_spans),
            }

            if "labels" in essay and essay["labels"]:
                truth_spans = bio_to_spans(essay["labels"], offsets)
                pred_indices = spans_to_index_set(payload["predicted_spans"])
                truth_indices = spans_to_index_set(truth_spans)
                doc_metrics = precision_recall_f1(pred_indices, truth_indices)
                payload["metrics"] = doc_metrics
                metrics.append(doc_metrics)

            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if metrics:
        mean_metrics = {
            key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]
        }
        print(
            "Mean char-level metrics "
            f"(n={len(metrics)} docs): {json.dumps(mean_metrics, indent=2)}"
        )
    else:
        print(
            "Inference finished on test set (no ground-truth labels available)."
        )


if __name__ == "__main__":
    main()

