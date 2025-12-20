#!/bin/bash
# Start MLflow Tracking Server with proper backend configuration

set -e

echo "🚀 Starting MLflow Tracking Server..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create directories if they don't exist
mkdir -p mlruns
mkdir -p mlartifacts

echo "📁 Backend Store: mlruns/mlflow.db"
echo "📦 Artifact Store: mlartifacts/"

# Start MLflow server with proper configuration (matching Docker setup)
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlruns/mlflow.db \
    --default-artifact-root ./mlartifacts

echo "✅ MLflow UI available at http://localhost:5000"
