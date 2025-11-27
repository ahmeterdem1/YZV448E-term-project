"""
Model utilities powering the PII cleaner.

The helpers in this module intentionally keep the business logic independent from
training code so they can be reused both during experimentation (e.g. notebooks)
and inside production services (e.g. FastAPI apps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch
import torch.nn as nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Pipeline,
    TokenClassificationPipeline,
    pipeline,
)

try:
    from vllm import LLM  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LLM = None  # type: ignore


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

    The class exposes ``mask_text`` and ``process_document`` helpers which are
    leveraged by :func:`get_batch_outputs`.
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
            device = 0 if torch.cuda.is_available() else -1

        self.pipeline: TokenClassificationPipeline = pipeline(
            task="token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=aggregation_strategy,
            device=device,
        )

    def forward(self, texts: Sequence[str]) -> List[List[EntitySpan]]:
        """
        Run inference on multiple texts and return detected entity spans.
        """

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
        """
        Replace each detected entity with the configured mask token.
        """

        spans = self.forward([text])[0]
        if not spans:
            return text

        masked_parts: List[str] = []
        cursor = 0
        for span in spans:
            masked_parts.append(text[cursor : span.start])
            masked_parts.append(self._replacement_for(span.label))
            cursor = span.end
        masked_parts.append(text[cursor:])
        return "".join(masked_parts)

    def process_document(self, document: str) -> str:
        """
        Alias method used by :func:`get_batch_outputs`.
        """

        return self.mask_text(document)


def _default_batch_handler(model: Any, document: str) -> str:
    """
    Gracefully fallback to different APIs depending on the model interface.
    """

    if hasattr(model, "process_document"):
        return model.process_document(document)

    if hasattr(model, "mask_text"):
        return model.mask_text(document)  # type: ignore[no-any-return]

    if callable(model):
        output = model(document)
        if isinstance(output, str):
            return output
        raise TypeError(
            "Callable model must return a string representing the masked document."
        )

    raise TypeError(
        "Model does not expose a known interface. Provide an object with either "
        "'process_document', 'mask_text' or make it callable."
    )


def get_batch_outputs(model: Any, documents: Mapping[str, str]) -> Dict[str, str]:
    """
    Process a batch of (document_name -> content) pairs.

    Parameters
    ----------
    model:
        Anything exposing ``process_document(str) -> str``, ``mask_text(str) -> str``
        or a callable returning the masked string directly.
    documents:
        Mapping of document identifiers to raw text.

    Returns
    -------
    dict
        Key/value pairs matching the provided documents with their cleaned content.
    """

    if not isinstance(documents, Mapping):
        raise TypeError("documents must be a mapping of name -> text.")

    cleaned: Dict[str, str] = {}
    for name, text in documents.items():
        if not isinstance(text, str):
            raise TypeError(f"Document '{name}' is not a string.")
        cleaned[name] = _default_batch_handler(model, text)
    return cleaned

def load_model(*args, **kwargs) -> HFPiiCleaner:
    raise NotImplementedError("Implement this function")
