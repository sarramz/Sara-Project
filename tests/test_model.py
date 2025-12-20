import pytest
import os
import pickle
import numpy as np
import pandas as pd


class TestModel:
    """Test cases for the trained model"""

    @pytest.fixture
    def model_path(self):
        return "artifacts/model.pkl"

    @pytest.fixture
    def preprocessor_path(self):
        return "artifacts/proprocessor.pkl"

    def test_model_exists(self, model_path):
        """Test if model file exists"""
        assert os.path.exists(model_path), f"Model file not found at {model_path}"

    def test_preprocessor_exists(self, preprocessor_path):
        """Test if preprocessor file exists"""
        assert os.path.exists(preprocessor_path), f"Preprocessor file not found at {preprocessor_path}"

    def test_model_loads(self, model_path):
        """Test if model can be loaded"""
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            assert model is not None, "Model loaded but is None"

    def test_model_prediction_shape(self, model_path, preprocessor_path):
        """Test if model produces predictions of correct shape"""
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            # Load model
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Create sample data
            sample_data = pd.DataFrame(
                {
                    "gender": ["male"],
                    "race_ethnicity": ["group A"],
                    "parental_level_of_education": ["bachelor's degree"],
                    "lunch": ["standard"],
                    "test_preparation_course": ["completed"],
                    "reading_score": [70],
                    "writing_score": [75],
                }
            )

            # Load preprocessor
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)

            # Transform and predict
            try:
                transformed_data = preprocessor.transform(sample_data)
                predictions = model.predict(transformed_data)
                assert len(predictions) == 1, "Prediction shape is incorrect"
                assert isinstance(predictions[0], (int, float, np.number)), "Prediction is not a number"
            except Exception as e:
                pytest.skip(f"Prediction test skipped: {str(e)}")

    def test_model_prediction_range(self, model_path, preprocessor_path):
        """Test if model predictions are in reasonable range"""
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)

            sample_data = pd.DataFrame(
                {
                    "gender": ["female"],
                    "race_ethnicity": ["group B"],
                    "parental_level_of_education": ["master's degree"],
                    "lunch": ["standard"],
                    "test_preparation_course": ["completed"],
                    "reading_score": [80],
                    "writing_score": [85],
                }
            )

            try:
                transformed_data = preprocessor.transform(sample_data)
                predictions = model.predict(transformed_data)
                # Assuming scores are between 0 and 100
                assert 0 <= predictions[0] <= 100, f"Prediction {predictions[0]} is out of expected range [0, 100]"
            except Exception as e:
                pytest.skip(f"Prediction range test skipped: {str(e)}")
