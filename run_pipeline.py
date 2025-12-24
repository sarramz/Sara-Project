"""
Complete MLOps Pipeline for Fake News Detection
Runs: Data Ingestion → Model Training (RoBERTa) → MLflow Tracking
"""

import os
import sys
from src.logger import logging
from src.exceptions import CustomException
from src.components.data_ingestion import DataIngestionFakeNews
from src.components.model_trainer import ModelTrainerRoBERTa


def run_fake_news_pipeline(
    data_path="data/fake_news.csv",
    sample_size=1000,
    model_name="roberta-base",
    epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    mlflow_uri="http://localhost:5000",
):
    """
    Run the complete fake news detection pipeline

    Args:
        data_path: Path to fake news CSV dataset
        sample_size: Limit dataset size (None = full dataset)
        model_name: HuggingFace model name (roberta-base, roberta-large)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        mlflow_uri: MLflow tracking URI

    Returns:
        f1_score: Final F1 score on test set
    """
    try:
        print("\n" + "=" * 70)
        print("   FAKE NEWS DETECTION - MLOps PIPELINE")
        print("=" * 70)
        print(f"Dataset: {data_path}")
        print(f"Model: {model_name}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"MLflow: {mlflow_uri}")
        if sample_size:
            print(f"Sample Size: {sample_size} (testing mode)")
        print("=" * 70 + "\n")

        # Step 1: Data Ingestion
        print("[1/2]  Starting Data Ingestion...")
        logging.info("Starting Data Ingestion")

        data_ingestion = DataIngestionFakeNews()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(
            data_path=data_path, sample_size=sample_size
        )

        print(f"✓ Data Ingestion Complete")
        print(f"  - Train: {train_data_path}")
        print(f"  - Test: {test_data_path}\n")

        # Step 2: Model Training
        print("[2/2] Starting Model Training (with MLflow tracking)...")
        logging.info("Starting Model Training")

        model_trainer = ModelTrainerRoBERTa()
        f1_score = model_trainer.initiate_model_trainer(
            train_path=train_data_path,
            test_path=test_data_path,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            mlflow_tracking_uri=mlflow_uri,
            experiment_name="fake_news_detection",
        )

        print(f" Model Training Complete\n")

        # Summary
        print("\n" + "=" * 70)
        print("   PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Model saved to: artifacts/roberta_fakenews/")
        print(f"\nNext steps:")
        print("  1. View experiments: mlflow ui --port 5000")
        print("  2. Test predictions: python src/Pipelines/predict_pipeline_fakenews.py")
        print("  3. Deploy API: python app_fakenews.py")
        print("=" * 70 + "\n")

        return f1_score

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fake News Detection Pipeline")
    parser.add_argument("--data", type=str, default="data/fake_news.csv", help="Path to fake news dataset CSV")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for quick testing (None = full dataset)")
    parser.add_argument(
        "--model", type=str, default="roberta-base", choices=["roberta-base", "roberta-large"], help="Model to use"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000", help="MLflow tracking URI")

    args = parser.parse_args()

    # Check if data file exists
    if not os.path.exists(args.data):
        print(f" Error: Dataset not found at {args.data}")
        print("\nPlease download a fake news dataset and place it at data/fake_news.csv")
        print("\nSuggested datasets:")
        print("  - WELFake: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification")
        print("  - Fake News Detection: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("\nDataset should have columns: 'text' and 'label' (optional: 'title')")
        sys.exit(1)

    # Run pipeline
    f1_score = run_fake_news_pipeline(
        data_path=args.data,
        sample_size=args.sample,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mlflow_uri=args.mlflow_uri,
    )

    print(f"\n Final F1 Score: {f1_score:.4f}")
