import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    TrainerCallback,
    PreTrainedTokenizerFast
)
from loguru import logger as app_logger

from app.core.config import settings
from app.utils.model import compute_f5_score

logger = logging.getLogger(__name__)


class LoguruCallback(TrainerCallback):
    """Intercepts Hugging Face logs and pipes them to Loguru."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Filter out internal keys to keep logs clean
            clean_logs = {k: v for k, v in logs.items() if k not in ["epoch", "step"]}
            epoch = logs.get("epoch", 0.0)
            step = state.global_step
            app_logger.info(f"📉 Trainer Log (Epoch {epoch:.2f} | Step {step}): {clean_logs}")


class TrainingService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.split_seed = 42  # Ensures the train/eval split is always the same

    def load_raw_dataset(self, file_path: str) -> Dataset:
        """Loads the Kaggle JSON format (list of dicts)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Kaggle dataset is a direct list of dicts.
            # If wrapped in "data" key (rare), handle it, otherwise assume list.
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

        return Dataset.from_list(data)

    def get_label_mappings(self, dataset: Dataset) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
        """
        Scans the dataset to find all unique string labels (BIO tags)
        and creates integer mappings.
        """
        unique_labels = set()

        # 'labels' is the key in Kaggle dataset
        if "labels" not in dataset.column_names:
            raise ValueError("Dataset missing 'labels' column. Cannot train without ground truth.")

        for ex in dataset:
            unique_labels.update(ex["labels"])

        label_list = sorted(list(unique_labels))
        label2id = {l: i for i, l in enumerate(label_list)}
        id2label = {i: l for i, l in enumerate(label_list)}

        return label2id, id2label, label_list

    def tokenize_and_align_labels(self, examples, tokenizer, label2id):
        """
        Tokenizes inputs and aligns string labels (BIO) to tokens.
        """
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True
        )

        labels = []
        for i, label_strs in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens (-100)
                if word_idx is None:
                    label_ids.append(-100)
                # Only label the first token of a word
                elif word_idx != previous_word_idx:
                    # Convert string label "B-NAME" to int ID
                    label_str = label_strs[word_idx]
                    label_ids.append(label2id.get(label_str, -100))
                else:
                    # Sub-words get -100 (ignored in loss)
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def train_model(self, base_model_path: str = "dslim/bert-base-NER") -> str:
        """
        Fine-tunes the model on train.json (using a 90/10 split internally).
        """
        app_logger.info("🚀 Starting model training process...")

        # 1. Load Data
        full_dataset = self.load_raw_dataset(settings.TRAIN_DATASET_PATH)

        # 2. Dynamic Label Mapping
        label2id, id2label, label_list = self.get_label_mappings(full_dataset)
        app_logger.info(f"🏷️ Found {len(label_list)} unique labels: {label_list}")

        # 3. Create Train/Eval Split (Deterministic)
        # We assume test.json has no labels, so we MUST split train.json
        dataset_split = full_dataset.train_test_split(test_size=0.1, seed=self.split_seed)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]

        app_logger.info(f"✂️ Data Split: {len(train_dataset)} Train | {len(eval_dataset)} Eval")

        # 4. Initialize Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_prefix_space=True)

        # Initialize model with correct number of labels
        model = AutoModelForTokenClassification.from_pretrained(
            base_model_path,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True  # Necessary if base model has different labels
        )

        # 5. Tokenize
        tokenized_train = train_dataset.map(
            lambda x: self.tokenize_and_align_labels(x, tokenizer, label2id),
            batched=True
        )
        tokenized_eval = eval_dataset.map(
            lambda x: self.tokenize_and_align_labels(x, tokenizer, label2id),
            batched=True
        )

        # 6. Training Args
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{settings.MODEL_REGISTRY_DIR}/model_{timestamp}"

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="no",  # We save manually at end
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            use_cpu=not torch.cuda.is_available(),
            logging_steps=10
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[LoguruCallback]
        )

        # 7. Train
        trainer.train()

        # 8. Save
        app_logger.info(f"💾 Saving new model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        return output_dir

    def evaluate_model(self, model_pipeline, dataset_path_ignored: str = None) -> Dict[str, float]:
        """
        Evaluates the model.
        NOTE: Since Kaggle 'test.json' has no labels, we ignore 'dataset_path_ignored'
        and load 'train.json' then deterministically split it to find the Eval set.
        """
        # Always use TRAIN_DATASET_PATH to derive the eval set
        source_path = settings.TRAIN_DATASET_PATH

        if not os.path.exists(source_path):
            app_logger.warning(f"⚠️ Source dataset not found at {source_path}")
            return {}

        app_logger.info("🧪 Preparing evaluation set from training data...")
        try:
            full_dataset = self.load_raw_dataset(source_path)

            # Re-create the same split as training
            dataset_split = full_dataset.train_test_split(test_size=0.1, seed=self.split_seed)
            eval_dataset = dataset_split["test"]

            # Label mapping must come from the MODEL config now, not the dataset
            # because the model is already trained and fixed.
            model = model_pipeline.model
            tokenizer = model_pipeline.tokenizer

            if not hasattr(model.config, "label2id"):
                app_logger.error("❌ Model config missing label2id. Cannot evaluate.")
                return {}

            label2id = model.config.label2id
            id2label = model.config.id2label
            label_list = list(id2label.values())

        except Exception as e:
            app_logger.error(f"❌ Failed to prepare eval dataset: {e}")
            return {}

        # Align
        try:
            tokenized_data = eval_dataset.map(
                lambda x: self.tokenize_and_align_labels(x, tokenizer, label2id),
                batched=True
            )
        except Exception as e:
            app_logger.error(f"❌ Tokenization failed during eval: {e}")
            return {}

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir="/tmp/eval_results",
            per_device_eval_batch_size=8,
            use_cpu=not torch.cuda.is_available(),
            logging_strategy="no"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[LoguruCallback]
        )

        app_logger.info(f"⏳ Running inference on {len(eval_dataset)} validation samples...")
        predictions_output = trainer.predict(tokenized_data)

        preds = np.argmax(predictions_output.predictions, axis=2)
        labels = predictions_output.label_ids

        metrics = compute_f5_score(preds, labels, label_list, beta=5.0)
        app_logger.success(f"📊 Evaluation Complete. F5 Score: {metrics.get('f5', 0.0):.4f}")

        return metrics