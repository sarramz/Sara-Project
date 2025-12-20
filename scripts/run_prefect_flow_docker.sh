#!/bin/bash
# Run Prefect Flow in Docker Compose environment
# This script executes the ML pipeline inside the Prefect container

set -e

echo "🚀 Running Prefect Flow in Docker Compose..."
echo

# Check if Docker Compose is running
if ! docker compose ps | grep -q "prefect_server"; then
    echo "❌ Error: Prefect container is not running!"
    echo
    echo "Please start Docker Compose first:"
    echo "  docker compose up -d"
    echo
    exit 1
fi

# Parse arguments (pass through to the Python script)
ARGS="$@"

# Default to quick test if no args provided
if [ -z "$ARGS" ]; then
    echo "ℹ️  No arguments provided. Running quick test (1000 samples, 1 epoch)"
    ARGS="--sample 1000 --epochs 1"
fi

echo "Running with arguments: $ARGS"
echo

# Execute the Prefect flow inside the container
docker compose exec prefect python /app/prefect_flows/ml_pipeline_flow.py $ARGS

echo
echo "✅ Prefect flow execution completed!"
echo
echo "View results:"
echo "  - Prefect UI: http://localhost:4200"
echo "  - MLflow UI: http://localhost:5000"
echo
