import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

from app.core.config import settings
from app.utils.model import compute_f5_score

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_dataset_from_json(self, file_path: str) -> Dataset:
        """Loads dataset from a JSON file (expecting 'tokens' and 'ner_tags' fields)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)
            # Ensure it's a list of dicts
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

        return Dataset.from_list(data)

    def train_model(self, base_model_path: str = "dslim/bert-base-NER") -> str:
        """
        Fine-tunes the model on the training dataset and saves it to the registry.
        Returns the path to the new model.
        """
        logger.info("🚀 Starting model training...")

        # Load Datasets
        try:
            train_dataset = self.load_dataset_from_json(settings.TRAIN_DATASET_PATH)
            # Use test set for validation during training if available
            eval_dataset = None
            if os.path.exists(settings.TEST_DATASET_PATH):
                eval_dataset = self.load_dataset_from_json(settings.TEST_DATASET_PATH)
        except Exception as e:
            logger.error(f"❌ Failed to load datasets: {e}")
            raise

        # Load Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        model = AutoModelForTokenClassification.from_pretrained(base_model_path)

        # Align labels
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"], truncation=True, is_split_into_words=True
            )
            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_and_align_labels, batched=True) if eval_dataset else None

        # Training Args
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{settings.MODEL_REGISTRY_DIR}/model_{timestamp}"

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch" if eval_dataset else "no",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="no",  # Save manually at the end
            use_cpu=not torch.cuda.is_available()
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

        # Save Model
        logger.info(f"💾 Saving new model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        return output_dir

    def evaluate_model(self, model_pipeline, dataset_path: str) -> Dict[str, float]:
        """
        Evaluates the loaded pipeline against a test dataset and returns F5 score.
        """
        if not os.path.exists(dataset_path):
            logger.warning(f"⚠️ Test dataset not found at {dataset_path}")
            return {}

        logger.info(f"🧪 Evaluating model on {dataset_path}")
        dataset = self.load_dataset_from_json(dataset_path)

        # We need to map the pipeline's string output back to IDs or
        # map the dataset's IDs to strings.
        # For simplicity, we assume the dataset has 'ner_tags' as IDs
        # and we use the model's config to map them.

        id2label = model_pipeline.model.config.id2label
        label_list = list(id2label.values())

        predictions = []
        true_labels = []

        # Run inference
        # Note: This is a simplified evaluation.
        # In a real scenario, we need to handle tokenization alignment meticulously.
        # Here we approximate by using the pipeline's output and matching text.

        # Easier approach for metric calculation: Use the raw model + tokenizer
        # and the compute_metrics logic, but we passed a pipeline wrapper.
        # We will access the underlying model/tokenizer from the wrapper.

        tokenizer = model_pipeline.tokenizer
        model = model_pipeline.model

        def tokenize_and_align(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"], truncation=True, is_split_into_words=True
            )
            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                prev_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != prev_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    prev_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_data = dataset.map(tokenize_and_align, batched=True)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Use a temporary Trainer just for evaluation
        training_args = TrainingArguments(
            output_dir="/tmp/eval",
            per_device_eval_batch_size=8,
            use_cpu=not torch.cuda.is_available()
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        predictions_output = trainer.predict(tokenized_data)
        preds = np.argmax(predictions_output.predictions, axis=2)
        labels = predictions_output.label_ids

        # Calculate F5
        metrics = compute_f5_score(preds, labels, label_list, beta=5.0)
        logger.info(f"📊 Evaluation Results: {metrics}")

        return metrics
