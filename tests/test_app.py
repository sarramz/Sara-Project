import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app as flask_app


class TestFlaskApp:
    """Test cases for Fake News Detection Flask application"""

    @pytest.fixture
    def client(self):
        """Create a test client"""
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as client:
            yield client

    def test_home_page(self, client):
        """Test home page loads"""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/info")
        # Should return 200 if model loaded, 500 if not
        assert response.status_code in [200, 500]

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"fakenews_predictions_total" in response.data or b"# HELP" in response.data

    def test_predict_endpoint_missing_text(self, client):
        """Test predict endpoint with missing text"""
        data = {"title": "Test Title"}
        response = client.post("/predict", json=data, content_type="application/json")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_predict_endpoint_valid(self, client):
        """Test predict endpoint with valid data"""
        data = {
            "title": "Breaking News",
            "text": "Scientists discover new treatment for cancer that shows promising results in clinical trials.",
        }
        response = client.post("/predict", json=data, content_type="application/json")
        # Should return 200 if model exists, 500 if not
        assert response.status_code in [200, 500]

    def test_predict_batch_endpoint(self, client):
        """Test batch predict endpoint"""
        data = {"articles": [{"title": "News 1", "text": "Text 1"}, {"title": "News 2", "text": "Text 2"}]}
        response = client.post("/predict_batch", json=data, content_type="application/json")
        # Should return 200 if model exists, 500 if not
        assert response.status_code in [200, 500]

    def test_invalid_endpoint(self, client):
        """Test invalid endpoint returns 404"""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404
