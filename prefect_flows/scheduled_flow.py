"""
Scheduled Prefect Flow for Automated Model Retraining (Prefect 3.x)
"""

from prefect import flow, task
import json
import os
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prefect_flows.ml_pipeline_flow import ml_pipeline_flow


@task(name="check_data_drift")
def check_data_drift_task():
    """
    Check if new data is available or data drift is detected
    
    Returns:
        bool: True if retraining is needed
    """
    # TODO: Implement actual data drift detection
    # For now, always return True to enable retraining
    
    # Example: Check if new data file exists
    new_data_marker = "data/new_data_available.flag"
    if os.path.exists(new_data_marker):
        os.remove(new_data_marker)  # Remove flag after reading
        return True
    
    # Example: Check last training time
    metadata_path = "artifacts/roberta_fakenews/metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        # Check if model is older than 7 days
        # last_trained = metadata.get('timestamp', datetime.now().isoformat())
        # ... implement time-based retraining logic
    
    return True  # Always retrain for now


@task(name="evaluate_model_performance")
def evaluate_model_performance_task(training_results: dict):
    """
    Evaluate if model performance meets threshold
    
    Args:
        training_results: Results from training
        
    Returns:
        bool: True if performance is acceptable
    """
    f1_threshold = 0.75  # Minimum acceptable F1 score
    
    f1_score = training_results['metrics']['f1']
    
    if f1_score >= f1_threshold:
        print(f"✅ Model performance acceptable: F1={f1_score:.4f} >= {f1_threshold}")
        return True
    else:
        print(f"⚠️ Model performance below threshold: F1={f1_score:.4f} < {f1_threshold}")
        return False


@task(name="notify_completion")
def notify_completion_task(pipeline_results: dict, performance_ok: bool):
    """
    Send notification about pipeline completion
    
    Args:
        pipeline_results: Results from pipeline
        performance_ok: Whether performance is acceptable
    """
    # TODO: Implement actual notification (email, Slack, etc.)
    
    message = f"""
    🤖 ML Pipeline Execution Report
    
    Status: {'✅ Success' if performance_ok else '⚠️ Warning'}
    Model: {pipeline_results['model_path']}
    
    Metrics:
    - F1 Score: {pipeline_results['metrics']['f1']:.4f}
    - Accuracy: {pipeline_results['metrics']['accuracy']:.4f}
    - Precision: {pipeline_results['metrics']['precision']:.4f}
    - Recall: {pipeline_results['metrics']['recall']:.4f}
    
    MLflow Run: {pipeline_results['mlflow_run_id']}
    """
    
    print(message)
    
    # Save report
    report_path = f"logs/pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("logs", exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(message)
    
    return report_path


@flow(name="scheduled_ml_pipeline")
def scheduled_ml_pipeline_flow(
    data_path: str = "data/fake_news.csv",
    check_drift: bool = True,
    model_name: str = "roberta-base",
    epochs: int = 3,
    batch_size: int = 8
):
    """
    Scheduled ML Pipeline with data drift checking
    
    Args:
        data_path: Path to data
        check_drift: Whether to check for data drift
        model_name: Model to train
        epochs: Training epochs
        batch_size: Batch size
    """
    # Check if retraining is needed
    if check_drift:
        should_retrain = check_data_drift_task()
        if not should_retrain:
            print("⏭️ Skipping training - no data drift detected")
            return {"status": "skipped", "reason": "no_drift"}
    
    # Run main pipeline
    pipeline_results = ml_pipeline_flow(
        data_path=data_path,
        sample_size=None,  # Use full dataset
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        mlflow_uri="http://localhost:5000"
    )
    
    # Evaluate performance
    performance_ok = evaluate_model_performance_task(pipeline_results)
    
    # Send notification
    report_path = notify_completion_task(pipeline_results, performance_ok)
    
    return {
        "status": "completed",
        "pipeline_results": pipeline_results,
        "performance_ok": performance_ok,
        "report_path": report_path
    }


if __name__ == "__main__":
    # Deploy with schedule using Prefect 3.x API
    scheduled_ml_pipeline_flow.deploy(
        name="ml-pipeline-daily-retrain",
        work_pool_name="default-agent-pool",
        tags=["ml", "scheduled", "retraining"],
        description="Scheduled ML pipeline with automatic retraining",
        version="1.0.0",
        parameters={
            "data_path": "data/fake_news.csv",
            "check_drift": True,
            "model_name": "roberta-base",
            "epochs": 3,
            "batch_size": 8
        },
        # Schedule: Run daily at 2 AM UTC
        cron="0 2 * * *",
    )
    
    print("\n" + "="*70)
    print("✅ SCHEDULED DEPLOYMENT CREATED")
    print("="*70)
    print(f"Deployment Name: ml-pipeline-daily-retrain")
    print(f"Schedule: Daily at 2:00 AM UTC")
    print("\n📋 The pipeline will automatically:")
    print("  1. Check for data drift")
    print("  2. Retrain model if needed")
    print("  3. Evaluate performance")
    print("  4. Send notifications")
    print("="*70 + "\n")