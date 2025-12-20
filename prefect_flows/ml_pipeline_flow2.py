"""
Prefect Flow for ML Pipeline
Orchestrates: Ingestion → Preparation → Validation → Training
"""

from prefect import flow, task

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.components.data_ingestion import DataIngestion
from src.components.data_preparation import DataPreparation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.logger import logging


@task(name="data_ingestion", retries=2, retry_delay_seconds=10)
def data_ingestion_task(data_path: str = "data/fake_news.csv", sample_size: int = None):
    """
    Task 1: Ingest raw data
    
    Args:
        data_path: Path to raw CSV file
        sample_size: Optional sample size for testing
        
    Returns:
        Path to raw data file
    """
    logging.info("🔄 Starting Data Ingestion Task")
    ingestion = DataIngestion()
    raw_data_path = ingestion.initiate_data_ingestion(
        data_path=data_path,
        sample_size=sample_size
    )
    logging.info(f"✅ Data Ingestion Complete: {raw_data_path}")
    return raw_data_path


@task(name="data_preparation", retries=2, retry_delay_seconds=10)
def data_preparation_task(raw_data_path: str):
    """
    Task 2: Clean and prepare data
    
    Args:
        raw_data_path: Path to raw data
        
    Returns:
        Path to prepared data file
    """
    logging.info("🔄 Starting Data Preparation Task")
    preparation = DataPreparation()
    prepared_data_path = preparation.initiate_data_preparation(raw_data_path)
    logging.info(f"✅ Data Preparation Complete: {prepared_data_path}")
    return prepared_data_path


@task(name="data_validation", retries=2, retry_delay_seconds=10)
def data_validation_task(prepared_data_path: str):
    """
    Task 3: Validate data and split into train/test
    
    Args:
        prepared_data_path: Path to prepared data
        
    Returns:
        Tuple of (train_path, test_path, report_path)
    """
    logging.info("🔄 Starting Data Validation Task")
    validation = DataValidation()
    train_path, test_path, report_path = validation.initiate_data_validation(
        prepared_data_path
    )
    logging.info(f"✅ Data Validation Complete")
    logging.info(f"   Train: {train_path}")
    logging.info(f"   Test: {test_path}")
    return train_path, test_path, report_path


@task(name="model_training", retries=1, retry_delay_seconds=30)
def model_training_task(
    train_path: str,
    test_path: str,
    model_name: str = "roberta-base",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    mlflow_uri: str = "http://localhost:5000"
):
    """
    Task 4: Train the model
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        model_name: HuggingFace model name
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        mlflow_uri: MLflow tracking URI
        
    Returns:
        Dictionary with training results
    """
    logging.info("🔄 Starting Model Training Task")
    
    config = ModelTrainerConfig(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mlflow_tracking_uri=mlflow_uri
    )
    
    trainer = ModelTrainer(config)
    results = trainer.initiate_model_trainer(train_path, test_path)
    
    logging.info(f"✅ Model Training Complete")
    logging.info(f"   Model: {results['model_path']}")
    logging.info(f"   F1 Score: {results['metrics']['f1']:.4f}")
    
    return results


@flow(
    name="ml_pipeline_complete",
    description="Complete ML Pipeline: Ingestion → Preparation → Validation → Training"
    
)
def ml_pipeline_flow(
    data_path: str = "data/fake_news.csv",
    sample_size: int = None,
    model_name: str = "roberta-base",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    mlflow_uri: str = "http://localhost:5000"
):
    """
    Complete ML Pipeline Flow
    
    Args:
        data_path: Path to raw CSV data
        sample_size: Optional sample size for testing
        model_name: HuggingFace model name
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        mlflow_uri: MLflow tracking URI
        
    Returns:
        Dictionary with all pipeline results
    """
    logging.info("="*70)
    logging.info("🚀 STARTING PREFECT ML PIPELINE")
    logging.info("="*70)
    
    # Task 1: Data Ingestion
    raw_data_path = data_ingestion_task(data_path, sample_size)
    
    # Task 2: Data Preparation
    prepared_data_path = data_preparation_task(raw_data_path)
    
    # Task 3: Data Validation
    train_path, test_path, report_path = data_validation_task(prepared_data_path)
    
    # Task 4: Model Training
    training_results = model_training_task(
        train_path=train_path,
        test_path=test_path,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mlflow_uri=mlflow_uri
    )
    
    # Collect all results
    pipeline_results = {
        "raw_data_path": raw_data_path,
        "prepared_data_path": prepared_data_path,
        "train_path": train_path,
        "test_path": test_path,
        "validation_report_path": report_path,
        "model_path": training_results["model_path"],
        "metrics": training_results["metrics"],
        "mlflow_run_id": training_results["mlflow_run_id"]
    }
    
    logging.info("="*70)
    logging.info("🎉 PREFECT ML PIPELINE COMPLETE")
    logging.info("="*70)
    logging.info(f"Model Path: {pipeline_results['model_path']}")
    logging.info(f"F1 Score: {pipeline_results['metrics']['f1']:.4f}")
    logging.info(f"Accuracy: {pipeline_results['metrics']['accuracy']:.4f}")
    logging.info("="*70)
    
    return pipeline_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Prefect ML Pipeline")
    parser.add_argument("--data", default="data/fake_news.csv", help="Path to data")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size")
    parser.add_argument("--model-name", default="roberta-base", help="Model name")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000", help="MLflow URI")
    
    args = parser.parse_args()
    
    # Run the flow
    result = ml_pipeline_flow(
        data_path=args.data,
        sample_size=args.sample_size,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mlflow_uri=args.mlflow_uri
    )
    
    print("\n✅ Pipeline execution complete!")
    print(f"Results: {result}")