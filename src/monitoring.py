"""
Model Monitoring Module
Tracks model performance metrics and exposes them for Prometheus
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from functools import wraps
import time


# Prometheus metrics
prediction_counter = Counter("model_predictions_total", "Total number of predictions made")

prediction_duration = Histogram("model_prediction_duration_seconds", "Time spent processing prediction")

prediction_score_gauge = Gauge("model_prediction_score", "Latest prediction score value")

error_counter = Counter("model_errors_total", "Total number of prediction errors")

feature_drift_gauge = Gauge("feature_drift_score", "Data drift score for input features", ["feature"])


class ModelMonitor:
    """Monitor model predictions and performance"""

    def __init__(self):
        self.predictions_history = []
        self.max_history = 1000

    def track_prediction(self, input_data, prediction, duration):
        """Track a single prediction"""
        prediction_counter.inc()
        prediction_duration.observe(duration)
        prediction_score_gauge.set(float(prediction))

        # Store for drift detection
        self.predictions_history.append({"input": input_data, "prediction": prediction, "timestamp": time.time()})

        # Keep only recent history
        if len(self.predictions_history) > self.max_history:
            self.predictions_history = self.predictions_history[-self.max_history :]

    def track_error(self, error_type):
        """Track prediction errors"""
        error_counter.inc()

    def calculate_drift(self, feature_name, current_value, reference_mean, reference_std):
        """Calculate simple z-score based drift"""
        if reference_std == 0:
            return 0
        z_score = abs((current_value - reference_mean) / reference_std)
        feature_drift_gauge.labels(feature=feature_name).set(z_score)
        return z_score

    def get_metrics(self):
        """Get current metrics in Prometheus format"""
        return generate_latest()


# Global monitor instance
monitor = ModelMonitor()


def monitor_prediction(func):
    """Decorator to monitor prediction functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            monitor.track_prediction(args[0] if args else None, result, duration)
            return result
        except Exception as e:
            monitor.track_error(type(e).__name__)
            raise

    return wrapper
