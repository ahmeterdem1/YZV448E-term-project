"""
Model utilities powering the PII cleaner.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
    pipeline,
)

try:
    from vllm import LLM  # type: ignore
except Exception:
    LLM = None


@dataclass
class EntitySpan:
    """Lightweight structure describing a detected entity span."""
    label: str
    start: int
    end: int
    score: float


class HFPiiCleaner(nn.Module):
    """
    Wrapper around a Hugging Face token-classification model that masks PII spans.
    """

    def __init__(
        self,
        model_name_or_path: str,
        aggregation_strategy: str = "simple",
        device: Optional[int] = None,
        confidence_threshold: float = 0.0,
        mask_token: str = "[PII]",
        label_replacements: Optional[Mapping[str, str]] = None,
        tokenizer_kwargs: Optional[Mapping[str, Any]] = None,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        import logging
        logger = logging.getLogger(__name__)

        super().__init__()
        self.mask_token = mask_token
        self.confidence_threshold = confidence_threshold
        self.label_replacements = dict(label_replacements or {})

        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **tokenizer_kwargs
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, **model_kwargs
        )

        if device is None:
            if torch.cuda.is_available():
                device = 0
                logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = -1
                logger.info("💻 Using CPU (GPU not available)")

        self.pipeline: TokenClassificationPipeline = pipeline(
            task="token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=aggregation_strategy,
            device=device,
        )

    def forward(self, texts: Sequence[str]) -> List[List[EntitySpan]]:
        outputs = self.pipeline(list(texts), batch_size=len(texts))
        if isinstance(outputs, dict):
            outputs = [outputs]

        spans: List[List[EntitySpan]] = []
        for sample in outputs:
            sample_spans: List[EntitySpan] = []
            for entity in sample:
                if entity["score"] < self.confidence_threshold:
                    continue
                sample_spans.append(
                    EntitySpan(
                        label=entity["entity_group"],
                        start=int(entity["start"]),
                        end=int(entity["end"]),
                        score=float(entity["score"]),
                    )
                )
            spans.append(sorted(sample_spans, key=lambda span: span.start))
        return spans

    def _replacement_for(self, label: str) -> str:
        return self.label_replacements.get(label, self.mask_token)

    def mask_text(self, text: str) -> str:
        """Original interface: returns just the cleaned string."""
        cleaned, _ = self.mask_text_with_stats(text)
        return cleaned

    def mask_text_with_stats(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Replace entities and return stats about what was found.
        Returns: (masked_text, dict_of_counts)
        """
        spans = self.forward([text])[0]
        stats = Counter()

        if not spans:
            return text, dict(stats)

        masked_parts: List[str] = []
        cursor = 0
        for span in spans:
            stats[span.label] += 1
            masked_parts.append(text[cursor : span.start])
            masked_parts.append(self._replacement_for(span.label))
            cursor = span.end
        masked_parts.append(text[cursor:])

        return "".join(masked_parts), dict(stats)

    def process_document(self, document: str) -> str:
        return self.mask_text(document)


def get_batch_outputs(
    model: Any,
    documents: Mapping[str, str]
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Process a batch and return both cleaned text and aggregated statistics.

    Returns:
    (
        {doc_id: cleaned_text},
        {label: total_count}
    )
    """

    if not isinstance(documents, Mapping):
        raise TypeError("documents must be a mapping of name -> text.")

    cleaned: Dict[str, str] = {}
    aggregated_stats = Counter()

    # Optimized path for HFPiiCleaner to get stats
    if isinstance(model, HFPiiCleaner):
        for name, text in documents.items():
            clean_text, doc_stats = model.mask_text_with_stats(text)
            cleaned[name] = clean_text
            aggregated_stats.update(doc_stats)
    else:
        # Fallback for generic models (no stats collection)
        for name, text in documents.items():
            if hasattr(model, "process_document"):
                cleaned[name] = model.process_document(text)
            elif hasattr(model, "mask_text"):
                cleaned[name] = model.mask_text(text)
            elif callable(model):
                cleaned[name] = model(text)
            else:
                 raise TypeError("Unknown model interface")

    return cleaned, dict(aggregated_stats)


def load_model(
    model_name: str = "dslim/bert-base-NER",
    confidence_threshold: float = 0.6,
    device: Optional[int] = None
) -> HFPiiCleaner:
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Loading PII model: {model_name}")
    try:
        model = HFPiiCleaner(
            model_name_or_path=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
            aggregation_strategy="simple"
        )
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
