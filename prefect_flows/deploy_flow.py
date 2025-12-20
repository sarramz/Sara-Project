"""
Deploy ML Pipeline to Prefect Server (Prefect 3.x)
Uses flow.deploy() method for deployment
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prefect_flows.ml_pipeline_flow import ml_pipeline_flow

if __name__ == "__main__":
    # Deploy using Prefect 3.x API
    ml_pipeline_flow.deploy(
        name="ml-pipeline-production",
        work_pool_name="default-agent-pool",  # Use your work pool name
        tags=["ml", "fake-news", "roberta", "production"],
        description="Complete ML pipeline for fake news detection",
        version="1.0.0",
        parameters={
            "data_path": "data/fake_news.csv",
            "sample_size": None,
            "model_name": "roberta-base",
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 2e-5,
            "mlflow_uri": "http://localhost:5000"
        },
        # Uncomment to add a schedule (daily at 2 AM UTC)
        # cron="0 2 * * *",
    )
    
    print("\n" + "="*70)
    print("✅ DEPLOYMENT SUCCESSFUL")
    print("="*70)
    print(f"Deployment Name: ml-pipeline-production")
    print(f"Flow: ml_pipeline_complete")
    print("\n📋 Next Steps:")
    print("1. View in Prefect UI:")
    print("   http://localhost:4200 (or http://prefect:4200)")
    print("\n2. Run the deployment:")
    print("   prefect deployment run 'ml_pipeline_complete/ml-pipeline-production'")
    print("\n3. Or trigger via UI")
    print("="*70 + "\n")