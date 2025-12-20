"""
Flask API for Fake News Detection
Serves predictions from trained RoBERTa model
Includes Prometheus monitoring
"""

from flask import Flask, request, render_template, jsonify, Response
from src.Pipelines.predict_pipeline import FakeNewsPredictionPipeline, CustomData
from src.exceptions import CustomException
from src.monitoring import monitor
from prometheus_client import generate_latest, Counter, Histogram, Gauge
import sys
import time

app = Flask(__name__)

# Prometheus metrics
prediction_counter = Counter("fakenews_predictions_total", "Total predictions made", ["prediction_type"])
prediction_duration = Histogram("fakenews_prediction_duration_seconds", "Prediction duration")
model_confidence = Gauge("fakenews_model_confidence", "Model prediction confidence")
api_requests = Counter("fakenews_api_requests_total", "Total API requests", ["endpoint", "method", "status"])
error_counter = Counter("fakenews_errors_total", "Total errors", ["error_type"])

# Initialize prediction pipeline
try:
    pipeline = FakeNewsPredictionPipeline(model_path="artifacts/roberta_fakenews")
    print("✓ Fake News Detection model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    print("Please train a model first: python run_pipeline_fakenews.py")
    pipeline = None


@app.route("/")
def index():
    """Home page"""
    return render_template("index_fakenews.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    start_time = time.time()

    try:
        api_requests.labels(endpoint="/predict", method="POST", status="attempt").inc()

        if pipeline is None:
            api_requests.labels(endpoint="/predict", method="POST", status="error").inc()
            error_counter.labels(error_type="model_not_loaded").inc()
            return jsonify({"error": "Model not loaded. Please train a model first.", "status": "error"}), 500

        # Get data from request
        if request.is_json:
            # JSON API request
            data = request.get_json()
            title = data.get("title", "")
            text = data.get("text", "")
        else:
            # Form request
            title = request.form.get("title", "")
            text = request.form.get("text", "")

        # Validate input
        if not text:
            api_requests.labels(endpoint="/predict", method="POST", status="error").inc()
            error_counter.labels(error_type="validation_error").inc()
            return jsonify({"error": "Text field is required", "status": "error"}), 400

        # Make prediction
        result = pipeline.predict_single(title=title, text=text)

        # Track metrics
        duration = time.time() - start_time
        prediction_duration.observe(duration)
        prediction_counter.labels(prediction_type=result["prediction"]).inc()
        model_confidence.set(result["confidence"])
        api_requests.labels(endpoint="/predict", method="POST", status="success").inc()

        # Return result
        response = {
            "prediction": result["prediction"],
            "confidence": f"{result['confidence']:.2%}",
            "probabilities": {
                "Real News": f"{result['probabilities']['Real News']:.2%}",
                "Fake News": f"{result['probabilities']['Fake News']:.2%}",
            },
            "status": "success",
        }

        return jsonify(response)

    except Exception as e:
        api_requests.labels(endpoint="/predict", method="POST", status="error").inc()
        error_counter.labels(error_type=type(e).__name__).inc()
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        if pipeline is None:
            return jsonify({"error": "Model not loaded", "status": "error"}), 500

        # Get articles from request
        data = request.get_json()
        articles = data.get("articles", [])

        if not articles:
            return jsonify({"error": "No articles provided", "status": "error"}), 400

        # Make predictions
        results = pipeline.predict_batch(articles)

        return jsonify({"results": results, "count": len(results), "status": "success"})

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": pipeline is not None,
            "model_path": "artifacts/roberta_fakenews" if pipeline else None,
        }
    )


@app.route("/info")
def info():
    """Model information endpoint"""
    try:
        if pipeline is None:
            return jsonify({"error": "Model not loaded", "status": "error"}), 500

        metadata = pipeline.metadata if pipeline.metadata else {}

        return jsonify(
            {
                "model_info": {
                    "model_name": metadata.get("model_name", "Unknown"),
                    "task": metadata.get("task", "fake_news_detection"),
                    "num_labels": metadata.get("num_labels", 2),
                    "label_names": metadata.get("label_names", ["Real News", "Fake News"]),
                    "max_length": metadata.get("max_length", 128),
                    "train_samples": metadata.get("train_samples", "N/A"),
                    "test_samples": metadata.get("test_samples", "N/A"),
                    "metrics": metadata.get("metrics", {}),
                },
                "status": "success",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), mimetype="text/plain")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  🚀 Fake News Detection API with Prometheus Monitoring")
    print("=" * 70)
    print("Endpoints:")
    print("  - GET  /           - Home page")
    print("  - POST /predict    - Single prediction")
    print("  - POST /predict_batch - Batch predictions")
    print("  - GET  /health     - Health check")
    print("  - GET  /info       - Model information")
    print("  - GET  /metrics    - Prometheus metrics")
    print("\nExample usage:")
    print("  curl -X POST http://localhost:8080/predict \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{")
    print('      "title": "Breaking News",')
    print('      "text": "Scientists discover new cancer treatment..."')
    print("    }'")
    print("\nMonitoring:")
    print("  - Metrics: http://localhost:8080/metrics")
    print("  - Prometheus: http://localhost:9090")
    print("  - Grafana: http://localhost:3000")
    print("=" * 70 + "\n")

    app.run(host="0.0.0.0", port=8080, debug=False)
