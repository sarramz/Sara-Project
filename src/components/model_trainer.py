"""
Model Trainer for RoBERTa Fake News Detection
Step 4: Train transformer model with validated data
"""

import os
import sys
import json
import torch
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.exceptions import CustomException
from src.logger import logging

# MLflow imports
import mlflow
import mlflow.transformers
import time
import urllib.request
from urllib.error import URLError

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

class FakeNewsDataset(Dataset):
    """PyTorch Dataset for fake news text"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "roberta_fakenews")
    model_name: str = "roberta-base"
    max_length: int = 128
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 2e-5
    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "fake_news_detection3"


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = None):
        self.config = config or ModelTrainerConfig()
        self.config.mlflow_tracking_uri = self.config.mlflow_tracking_uri.rstrip("/")
        logging.info("Model Trainer initialized")

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def check_mlflow_connection(self, max_retries=3, retry_delay=2):
        mlflow_uri = self.config.mlflow_tracking_uri

        for attempt in range(max_retries):
            try:
                urllib.request.urlopen(f"{mlflow_uri}/health", timeout=5)
                logging.info(f"MLflow server connected: {mlflow_uri}")
                return True
            except (URLError, Exception) as e:
                if attempt < max_retries - 1:
                    logging.warning(
                        f"MLflow connection attempt {attempt + 1}/{max_retries} failed. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logging.warning(f"MLflow server not accessible at {mlflow_uri}")
                    logging.warning(f"Error: {str(e)}")
                    return False
        return False

    def initiate_model_trainer(self, train_path, test_path):
        logging.info("=" * 70)
        logging.info("STEP 4: MODEL TRAINING")
        logging.info("=" * 70)

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_texts = train_df["text"].tolist()
            train_labels = train_df["label"].tolist()
            test_texts = test_df["text"].tolist()
            test_labels = test_df["label"].tolist()

            tokenizer = RobertaTokenizer.from_pretrained(self.config.model_name)
            model = RobertaForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=2
            )

            train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer, self.config.max_length)
            test_dataset = FakeNewsDataset(test_texts, test_labels, tokenizer, self.config.max_length)

            output_dir = self.config.trained_model_path
            os.makedirs(output_dir, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.config.epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                warmup_steps=50,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=10,
                save_total_limit=1,
                report_to="none"
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=self.compute_metrics,
            )

            mlflow_enabled = self.check_mlflow_connection()

            mlflow_run_id = None

            if mlflow_enabled:
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
                mlflow.set_experiment(self.config.experiment_name)

                with mlflow.start_run(
                    run_name=f"{self.config.model_name}_epochs_{self.config.epochs}"
                ) as run:
                    mlflow.log_param("model_name", self.config.model_name)
                    mlflow.log_param("epochs", self.config.epochs)
                    mlflow.log_param("batch_size", self.config.batch_size)
                    mlflow.log_param("learning_rate", self.config.learning_rate)

                    trainer.train()
                    test_results = trainer.evaluate(test_dataset)

                    mlflow.log_metrics({
                        "accuracy": test_results["eval_accuracy"],
                        "precision": test_results["eval_precision"],
                        "recall": test_results["eval_recall"],
                        "f1": test_results["eval_f1"],
                        "loss": test_results["eval_loss"],
                    })

                    # Save model locally first
                    trainer.save_model(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    # Log model artifacts without using model registry
                    try:
                        # Log the entire model directory as an artifact
                        mlflow.log_artifacts(output_dir, artifact_path="model")
                        logging.info("✓ Model artifacts logged to MLflow")
                    except Exception as e:
                        logging.warning(f"Could not log model artifacts: {str(e)}")
                        logging.info("Model is still saved locally at: {output_dir}")

                    mlflow_run_id = run.info.run_id
                    logging.info(f"✓ MLflow Run ID: {mlflow_run_id}")
            else:
                trainer.train()
                test_results = trainer.evaluate(test_dataset)
                trainer.save_model(output_dir)
                tokenizer.save_pretrained(output_dir)

            logging.info(f"✓ Model saved locally: {output_dir}")

            return {
                "model_path": output_dir,
                "metrics": {
                    "accuracy": test_results["eval_accuracy"],
                    "precision": test_results["eval_precision"],
                    "recall": test_results["eval_recall"],
                    "f1": test_results["eval_f1"],
                },
                "mlflow_run_id": mlflow_run_id,
            }

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Training Step")
    parser.add_argument(
        "--train-data",
        default="artifacts/train.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--test-data",
        default="artifacts/test.csv",
        help="Path to test data CSV"
    )
    parser.add_argument(
        "--model-name",
        default="roberta-base",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5000/",
        help="MLflow tracking URI"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.train_data):
        print(f" Training data not found: {args.train_data}")
        print("Run data validation first: python -m src.components.data_validation")
        sys.exit(1)
    
    if not os.path.exists(args.test_data):
        print(f" Test data not found: {args.test_data}")
        print("Run data validation first: python -m src.components.data_validation")
        sys.exit(1)
    
    config = ModelTrainerConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mlflow_tracking_uri=args.mlflow_uri
    )
    
    trainer = ModelTrainer(config)
    results = trainer.initiate_model_trainer(args.train_data, args.test_data)
    
    print(f"\n Model training completed!")
    print(f"Model saved to: {results['model_path']}")
    print(f"F1 Score: {results['metrics']['f1']:.4f}")