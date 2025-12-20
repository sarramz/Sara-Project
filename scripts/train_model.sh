#!/bin/bash
# Train Model with MLflow Tracking

set -e  # Exit on error

echo "🚀 Starting ML Training Pipeline..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Run the complete pipeline
python run_pipeline.py

echo ""
echo "✅ Training completed!"
echo ""
echo "Next steps:"
echo "  1. View experiments: http://localhost:5000"
echo "  2. Version artifacts: dvc add artifacts/model.pkl artifacts/proprocessor.pkl"
echo "  3. Commit changes: git add . && git commit -m 'Update model'"
