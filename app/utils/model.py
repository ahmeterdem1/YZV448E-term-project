"""
Model utilities powering the PII cleaner.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import re

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


class RegexPatterns:
    """Collection of regex patterns for common PII types."""
    
    # Email pattern: matches most email formats
    EMAIL = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # US Phone numbers: (123) 456-7890, 123-456-7890, 1234567890
    PHONE_US = r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
    
    # International phone: +1-234-567-8900 format
    PHONE_INTL = r'\+\d{1,3}[-.\s]?\d{1,14}\b'
    
    # Social Security Number: XXX-XX-XXXX
    SSN = r'\b\d{3}-\d{2}-\d{4}\b'
    
    # Credit Card: 16 digits with optional spaces/dashes
    CREDIT_CARD = r'\b(?:\d[-\s]*){13}(?:\d[-\s]*)\b'
    
    # US ZIP Code: 5 digits or 5+4 format
    ZIP_CODE = r'\b\d{5}(?:-\d{4})?\b'
    
    # IPv4 Address
    IPV4 = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    
    # URLs: http(s), ftp, www
    URL = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    
    # Credit Card Expiry: MM/YY or MM/YYYY
    CARD_EXPIRY = r'\b(?:0[1-9]|1[0-2])/(?:\d{2}|\d{4})\b'
    
    # Student ID / Employee ID: Common patterns
    ID_PATTERN = r'\b(?:ID|SNO|REF)[-#]?:?\s*([A-Z0-9\-]{4,12})\b'
    
    # Passport Number: Usually 6-9 characters
    PASSPORT = r'\b[A-Z]{1,2}\d{6,8}\b'


class RegexPiiCleaner:
    """
    Detects PII using regex patterns.
    Returns EntitySpan objects compatible with HFPiiCleaner.
    """
    
    def __init__(self, mask_token: str = "[PII]"):
        self.mask_token = mask_token
        self.patterns = {
            'EMAIL': RegexPatterns.EMAIL,
            'PHONE': RegexPatterns.PHONE_INTL,  # Uses international pattern (more reliable)
            'SSN': RegexPatterns.SSN,
            'CREDIT_CARD': RegexPatterns.CREDIT_CARD,
            'ZIP_CODE': RegexPatterns.ZIP_CODE,
            'URL': RegexPatterns.URL,
            'IPV4': RegexPatterns.IPV4,
            'CARD_EXPIRY': RegexPatterns.CARD_EXPIRY,
            'ID': RegexPatterns.ID_PATTERN,
            'PASSPORT': RegexPatterns.PASSPORT,
        }
    
    def extract_spans(self, text: str) -> List[EntitySpan]:
        """Find all PII matches in text and return as EntitySpan objects."""
        spans: List[EntitySpan] = []
        
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                span = EntitySpan(
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    score=0.95  # High confidence for regex matches
                )
                spans.append(span)
        
        # Sort by start position and remove overlaps
        return self._merge_overlapping_spans(sorted(spans, key=lambda x: x.start))
    
    @staticmethod
    def _merge_overlapping_spans(spans: List[EntitySpan]) -> List[EntitySpan]:
        """Remove overlapping spans, keeping the first one."""
        if not spans:
            return []
        
        merged = [spans[0]]
        for current in spans[1:]:
            last = merged[-1]
            # If current span overlaps with last span, skip it
            if current.start < last.end:
                continue
            merged.append(current)
        
        return merged
    
    def mask_text(self, text: str) -> str:
        """Replace all detected PII with mask token."""
        cleaned, _ = self.mask_text_with_stats(text)
        return cleaned
    
    def mask_text_with_stats(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Replace PII and return statistics."""
        spans = self.extract_spans(text)
        stats = Counter()
        
        if not spans:
            return text, dict(stats)
        
        masked_parts = []
        cursor = 0
        
        for span in spans:
            stats[span.label] += 1
            masked_parts.append(text[cursor:span.start])
            masked_parts.append(self.mask_token)
            cursor = span.end
        
        masked_parts.append(text[cursor:])
        return "".join(masked_parts), dict(stats)


class HybridPiiCleaner:
    """
    Combines BERT model + Regex patterns for comprehensive PII detection.
    Uses both methods and merges results intelligently.
    """
    
    def __init__(
        self,
        bert_model: HFPiiCleaner,
        use_regex: bool = True,
        confidence_threshold: float = 0.6
    ):
        self.bert_model = bert_model
        self.regex_cleaner = RegexPiiCleaner() if use_regex else None
        self.confidence_threshold = confidence_threshold
    
    def _merge_spans(self, bert_spans: List[EntitySpan], regex_spans: List[EntitySpan]) -> List[EntitySpan]:
        """Merge BERT and Regex spans, removing duplicates."""
        all_spans = bert_spans + regex_spans
        
        if not all_spans:
            return []
        
        # Sort by start position
        all_spans.sort(key=lambda x: x.start)
        
        # Remove overlapping spans (keep BERT results if overlap)
        merged = [all_spans[0]]
        bert_span_set = {(s.start, s.end) for s in bert_spans}
        
        for current in all_spans[1:]:
            last = merged[-1]
            
            # Skip if overlaps with last span
            if current.start < last.end:
                continue
            
            merged.append(current)
        
        return merged
    
    def mask_text_with_stats(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Use both BERT and Regex, merge results, and return masked text + stats.
        """
        # Get BERT detections
        bert_spans = self.bert_model.forward([text])[0]
        bert_spans = [s for s in bert_spans if s.score >= self.confidence_threshold]
        
        # Get Regex detections
        regex_spans = []
        if self.regex_cleaner:
            regex_spans = self.regex_cleaner.extract_spans(text)
        
        # Merge all spans
        all_spans = self._merge_spans(bert_spans, regex_spans)
        stats = Counter()
        
        if not all_spans:
            return text, dict(stats)
        
        # Build masked text
        masked_parts = []
        cursor = 0
        
        for span in all_spans:
            stats[span.label] += 1
            masked_parts.append(text[cursor:span.start])
            replacement = self.bert_model._replacement_for(span.label)
            masked_parts.append(replacement)
            cursor = span.end
        
        masked_parts.append(text[cursor:])
        return "".join(masked_parts), dict(stats)
    
    def mask_text(self, text: str) -> str:
        """Simple interface."""
        cleaned, _ = self.mask_text_with_stats(text)
        return cleaned
    
    def forward(self, texts: Sequence[str]) -> List[List[EntitySpan]]:
        """Process multiple texts."""
        results = []
        for text in texts:
            bert_spans = self.bert_model.forward([text])[0]
            bert_spans = [s for s in bert_spans if s.score >= self.confidence_threshold]
            
            regex_spans = []
            if self.regex_cleaner:
                regex_spans = self.regex_cleaner.extract_spans(text)
            
            merged = self._merge_spans(bert_spans, regex_spans)
            results.append(merged)
        
        return results


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

    # Support for HybridPiiCleaner
    if isinstance(model, HybridPiiCleaner):
        for name, text in documents.items():
            clean_text, doc_stats = model.mask_text_with_stats(text)
            cleaned[name] = clean_text
            aggregated_stats.update(doc_stats)
    # Optimized path for HFPiiCleaner to get stats
    elif isinstance(model, HFPiiCleaner):
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
    device: Optional[int] = None,
    use_hybrid: bool = True
) -> Any:
    """
    Load PII detection model.
    
    Args:
        model_name: HuggingFace model name
        confidence_threshold: BERT confidence threshold
        device: GPU device ID or -1 for CPU
        use_hybrid: If True, combine BERT + Regex. If False, use only BERT.
    
    Returns:
        HybridPiiCleaner or HFPiiCleaner depending on use_hybrid flag
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Loading PII model: {model_name}")
    try:
        bert_model = HFPiiCleaner(
            model_name_or_path=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
            aggregation_strategy="simple"
        )
        logger.info("BERT model loaded successfully")
        
        if use_hybrid:
            hybrid_model = HybridPiiCleaner(bert_model, use_regex=True)
            logger.info("✅ Hybrid PII Cleaner (BERT + Regex) initialized")
            return hybrid_model
        else:
            logger.info("✅ BERT-only PII Cleaner initialized")
            return bert_model
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise