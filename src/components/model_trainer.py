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
import mlflow.pytorch
import time
import urllib.request
from urllib.error import URLError


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
    """Configuration for model training"""
    trained_model_path: str = os.path.join("artifacts", "roberta_fakenews")
    model_name: str = "roberta-base"
    max_length: int = 128
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "fake_news_detection3"


class ModelTrainer:
    """Train RoBERTa model for fake news detection"""

    def __init__(self, config: ModelTrainerConfig = None):
        self.config = config or ModelTrainerConfig()
        logging.info("Model Trainer initialized")

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
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
        """
        Check if MLflow server is accessible with retries

        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Seconds to wait between retries

        Returns:
            bool: True if connected, False otherwise
        """
        mlflow_uri = self.config.mlflow_tracking_uri
        
        for attempt in range(max_retries):
            try:
                health_url = f"{mlflow_uri}/health"
                urllib.request.urlopen(health_url, timeout=5)
                logging.info(f" MLflow server connected: {mlflow_uri}")
                return True
            except (URLError, Exception) as e:
                if attempt < max_retries - 1:
                    logging.warning(
                        f"  MLflow connection attempt {attempt + 1}/{max_retries} failed. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logging.warning(f"  MLflow server not accessible at {mlflow_uri}")
                    logging.warning(f"   Error: {str(e)}")
                    return False
        return False

    def log_model_to_mlflow(self, model, tokenizer, output_dir):
        """
        Log model to MLflow with fallback for compatibility

        Args:
            model: Trained model
            tokenizer: Tokenizer
            output_dir: Directory where model is saved

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logging.info("Attempting to log model using mlflow.pytorch.log_model...")
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=None
            )
            logging.info(" Model logged to MLflow using log_model")
            return True

        except Exception as log_error:
            logging.warning(f"  Could not use log_model: {str(log_error)}")
            logging.info("Trying fallback method: logging model as artifact...")

            try:
                import tempfile

                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Save model state dict
                    model_path = os.path.join(tmp_dir, "pytorch_model.bin")
                    torch.save(model.state_dict(), model_path)
                    mlflow.log_artifact(model_path, artifact_path="model")
                    logging.info("Model state dict logged as artifact")

                    # Save model config
                    config_path = os.path.join(tmp_dir, "config.json")
                    model.config.to_json_file(config_path)
                    mlflow.log_artifact(config_path, artifact_path="model")
                    logging.info(" Model config logged as artifact")

                    # Save tokenizer
                    tokenizer_dir = os.path.join(tmp_dir, "tokenizer")
                    os.makedirs(tokenizer_dir, exist_ok=True)
                    tokenizer.save_pretrained(tokenizer_dir)
                    mlflow.log_artifacts(tokenizer_dir, artifact_path="model/tokenizer")
                    logging.info(" Tokenizer logged as artifact")

                    # Log model metadata
                    mlflow.log_dict(
                        {
                            "model_type": "roberta",
                            "model_class": model.__class__.__name__,
                            "num_labels": model.config.num_labels,
                            "vocab_size": model.config.vocab_size,
                            "hidden_size": model.config.hidden_size,
                        },
                        "model/model_info.json",
                    )
                    logging.info(" Model metadata logged")

                return True

            except Exception as fallback_error:
                logging.error(f" Fallback method also failed: {str(fallback_error)}")
                return False

    def initiate_model_trainer(self, train_path, test_path):
        """
        Train RoBERTa model for fake news detection

        Args:
            train_path: Path to training CSV
            test_path: Path to test CSV

        Returns:
            Dictionary with training results
        """
        logging.info("=" * 70)
        logging.info("STEP 4: MODEL TRAINING")
        logging.info("=" * 70)

        try:
            # Load data
            logging.info(f"Loading data from {train_path} and {test_path}")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_texts = train_df["text"].tolist()
            train_labels = train_df["label"].tolist()
            test_texts = test_df["text"].tolist()
            test_labels = test_df["label"].tolist()

            logging.info(f"Train samples: {len(train_texts)}")
            logging.info(f"Test samples: {len(test_texts)}")

            # Initialize tokenizer and model
            logging.info(f"Loading {self.config.model_name}...")
            tokenizer = RobertaTokenizer.from_pretrained(self.config.model_name)
            model = RobertaForSequenceClassification.from_pretrained(
                self.config.model_name, 
                num_labels=2
            )

            # Create datasets
            logging.info("Creating datasets...")
            train_dataset = FakeNewsDataset(
                train_texts, train_labels, tokenizer, self.config.max_length
            )
            test_dataset = FakeNewsDataset(
                test_texts, test_labels, tokenizer, self.config.max_length
            )

            # Training arguments
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
                report_to="none",
                eval_strategy="epoch",
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=self.compute_metrics,
            )

            # MLflow tracking with connection check
            mlflow_enabled = self.check_mlflow_connection(max_retries=3, retry_delay=2)

            if mlflow_enabled:
                logging.info(" MLflow tracking: ENABLED")
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
                mlflow.set_experiment(self.config.experiment_name)
            else:
                logging.warning(" MLflow tracking: DISABLED (server not accessible)")
                logging.warning("   Training will continue without MLflow logging")

            # Start MLflow run if enabled
            mlflow_context = None
            if mlflow_enabled:
                mlflow_context = mlflow.start_run(
                    run_name=f"{self.config.model_name}_epochs_{self.config.epochs}"
                )
                logging.info(f"   MLflow tracking URI: {mlflow.get_tracking_uri()}")
                mlflow_context.__enter__()

            try:
                logging.info(" Starting training...")

                # Log parameters to MLflow if enabled
                if mlflow_enabled:
                    mlflow.log_param("model_name", self.config.model_name)
                    mlflow.log_param("task", "fake_news_detection")
                    mlflow.log_param("max_length", self.config.max_length)
                    mlflow.log_param("batch_size", self.config.batch_size)
                    mlflow.log_param("epochs", self.config.epochs)
                    mlflow.log_param("learning_rate", self.config.learning_rate)
                    mlflow.log_param("train_samples", len(train_texts))
                    mlflow.log_param("test_samples", len(test_texts))

                # Train
                trainer.train()
                logging.info(" Training completed!")

                # Evaluate on test set
                logging.info(" Evaluating on test set...")
                test_results = trainer.evaluate(test_dataset)

                # Log metrics to MLflow if enabled
                if mlflow_enabled:
                    mlflow.log_metric("accuracy", test_results["eval_accuracy"])
                    mlflow.log_metric("precision", test_results["eval_precision"])
                    mlflow.log_metric("recall", test_results["eval_recall"])
                    mlflow.log_metric("f1_score", test_results["eval_f1"])
                    mlflow.log_metric("test_loss", test_results["eval_loss"])

                    # Log model
                    logging.info("Logging model to MLflow...")
                    model_logged = self.log_model_to_mlflow(model, tokenizer, output_dir)

                    if not model_logged:
                        logging.warning("⚠️  Model could not be logged to MLflow")

                    mlflow_run_id = mlflow.active_run().info.run_id
                    logging.info(f" MLflow Run ID: {mlflow_run_id}")
                else:
                    mlflow_run_id = None

                # Save model locally
                logging.info(f"Saving model to {output_dir}...")
                trainer.save_model(output_dir)
                tokenizer.save_pretrained(output_dir)

                # Save metadata inside model directory
                metadata = {
                    "model_type": "roberta",
                    "model_name": self.config.model_name,
                    "task": "fake_news_detection",
                    "num_labels": 2,
                    "label_names": ["Real News", "Fake News"],
                    "max_length": self.config.max_length,
                    "train_samples": len(train_texts),
                    "test_samples": len(test_texts),
                    "training_config": {
                        "batch_size": self.config.batch_size,
                        "epochs": self.config.epochs,
                        "learning_rate": self.config.learning_rate,
                    },
                    "metrics": {
                        "accuracy": float(test_results["eval_accuracy"]),
                        "precision": float(test_results["eval_precision"]),
                        "recall": float(test_results["eval_recall"]),
                        "f1": float(test_results["eval_f1"]),
                        "loss": float(test_results["eval_loss"]),
                    },
                    "mlflow_run_id": mlflow_run_id,
                    "mlflow_tracking_uri": self.config.mlflow_tracking_uri if mlflow_enabled else None,
                }

                with open(f"{output_dir}/metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                # Save metrics separately for DVC (outside model directory)
                metrics_for_dvc = {
                    "accuracy": float(test_results["eval_accuracy"]),
                    "precision": float(test_results["eval_precision"]),
                    "recall": float(test_results["eval_recall"]),
                    "f1": float(test_results["eval_f1"]),
                    "loss": float(test_results["eval_loss"]),
                }

                with open("artifacts/metrics.json", "w") as f:
                    json.dump(metrics_for_dvc, f, indent=2)

                # Save label mapping
                label_map = {"0": "Real News", "1": "Fake News"}
                with open(f"{output_dir}/label_map.json", "w") as f:
                    json.dump(label_map, f, indent=2)

                logging.info("=" * 70)
                logging.info(" MODEL TRAINING COMPLETE!")
                logging.info("=" * 70)
                logging.info(f"Model: {self.config.model_name}")
                logging.info(f"Accuracy:  {test_results['eval_accuracy']:.4f}")
                logging.info(f"Precision: {test_results['eval_precision']:.4f}")
                logging.info(f"Recall:    {test_results['eval_recall']:.4f}")
                logging.info(f"F1 Score:  {test_results['eval_f1']:.4f}")
                logging.info(f"Model saved: {output_dir}")
                if mlflow_enabled and mlflow_run_id:
                    logging.info(f"MLflow Run: {mlflow_run_id}")
                logging.info("=" * 70)

                return {
                    "model_path": output_dir,
                    "metrics": metadata["metrics"],
                    "mlflow_run_id": mlflow_run_id,
                }

            finally:
                # Close MLflow run if it was started
                if mlflow_enabled and mlflow_context:
                    try:
                        mlflow_context.__exit__(None, None, None)
                        logging.info(" MLflow run closed")
                    except Exception as e:
                        logging.warning(f"Warning: Error closing MLflow run: {str(e)}")

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
        print(f"Test data not found: {args.test_data}")
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