"""
Simple deployment script - Run this ONCE inside the container
"""
import sys
sys.path.insert(0, '/app')

from prefect_flows.ml_pipeline_flow import ml_pipeline_flow
from prefect.deployments import Deployment

# Create deployment
deployment = Deployment.build_from_flow(
    flow=ml_pipeline_flow,
    name="ml-pipeline",
    work_pool_name="default-agent-pool",
    parameters={
        "data_path": "/app/data/fake_news.csv",
        "sample_size": 2000,
        "model_name": "roberta-base",
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "mlflow_uri": "http://mlflow:5000"
    }
)

deployment.apply()
print("✓ Deployment created!")
print("Run with: prefect deployment run 'ml_pipeline_complete/ml-pipeline'")