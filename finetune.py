"""
Optional fine-tuning of cardiffnlp/twitter-roberta-base-sentiment
on an IPL-specific tweet dataset.

Usage:
    python finetune.py --dataset data/ipl_tweets.csv --output models/ipl-roberta

Dataset CSV format:
    text,label
    "Kohli hits a six!",Positive
    "Wicket falls, disappointing",Negative
    "End of over 15",Neutral
"""
import argparse
import os
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def load_dataset(csv_path: str):
    texts, labels = [], []
    label_map = {"Positive": 2, "Neutral": 1, "Negative": 0}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("text") and row.get("label") in label_map:
                texts.append(row["text"])
                labels.append(label_map[row["label"]])
    return texts, labels


def run_finetune(dataset_path: str, output_dir: str,
                 epochs: int = 3, batch_size: int = 16,
                 lr: float = 2e-5):
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer,
        )
        from torch.utils.data import Dataset as TorchDataset
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Run: pip install transformers torch")
        return

    BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    logger.info(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, ignore_mismatched_sizes=True)

    logger.info(f"Loading dataset from {dataset_path}")
    texts, labels = load_dataset(dataset_path)
    logger.info(f"  {len(texts)} samples loaded")

    class IPLDataset(TorchDataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, truncation=True, padding=True,
                                 max_length=128, return_tensors="pt")
            self.labels = torch.tensor(labels)

        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()} | {"labels": self.labels[i]}

    # 80/20 split
    split = int(0.8 * len(texts))
    train_ds = IPLDataset(texts[:split], labels[:split])
    eval_ds  = IPLDataset(texts[split:], labels[split:])

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    logger.info("Starting fine-tuning …")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    logger.info("To use the fine-tuned model, set MODEL_PATH env var:")
    logger.info(f"  export MODEL_PATH={output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune IPL sentiment model")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    parser.add_argument("--output",  default="models/ipl-roberta")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    args = parser.parse_args()

    run_finetune(args.dataset, args.output, args.epochs, args.batch_size, args.lr)
