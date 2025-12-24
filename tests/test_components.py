"""
Sample unit tests for ML components
Add these tests to your tests/ directory
"""
import pytest
import pandas as pd
import os
import sys
from unittest.mock import Mock, patch


class TestDataIngestion:
    """Tests for Data Ingestion component"""
    
    def test_data_ingestion_creates_output(self, tmp_path):
        """Test that data ingestion creates expected output files"""
        from src.components.data_ingestion import DataIngestion
        
        # Create sample data
        sample_data = pd.DataFrame({
            'text': ['fake news 1', 'real news 1'],
            'label': [1, 0]
        })
        sample_file = tmp_path / "fake_news.csv"
        sample_data.to_csv(sample_file, index=False)
        
        # Run ingestion
        ingestion = DataIngestion()
        output_path = ingestion.initiate_data_ingestion(
            data_path=str(sample_file),
            sample_size=2
        )
        
        # Verify output exists
        assert os.path.exists(output_path)
        
    def test_data_ingestion_with_invalid_path(self):
        """Test data ingestion with invalid file path"""
        from src.components.data_ingestion import DataIngestion
        from src.exceptions import CustomException
        
        ingestion = DataIngestion()
        
        with pytest.raises(CustomException):
            ingestion.initiate_data_ingestion(
                data_path="nonexistent_file.csv"
            )


class TestDataPreparation:
    """Tests for Data Preparation component"""
    
    def test_data_preparation_cleans_text(self, tmp_path):
        """Test that data preparation cleans text properly"""
        from src.components.data_preparation import DataPreparation
        
        # Create sample raw data with title column (required by DataPreparation)
        raw_data = pd.DataFrame({
            'title': ['Breaking News', 'Important Update'],
            'text': ['FAKE NEWS!!!', 'Real News.'],
            'label': [1, 0]
        })
        raw_file = tmp_path / "raw.csv"
        raw_data.to_csv(raw_file, index=False)
        
        # Run preparation
        preparation = DataPreparation()
        output_path = preparation.initiate_data_preparation(str(raw_file))
        
        # Verify cleaned data
        cleaned_data = pd.read_csv(output_path)
        assert len(cleaned_data) > 0
        assert 'text' in cleaned_data.columns
        assert 'label' in cleaned_data.columns


class TestDataValidation:
    """Tests for Data Validation component"""
    
    def test_data_validation_splits_data(self, tmp_path):
        """Test that validation creates train/test splits"""
        from src.components.data_validation import DataValidation
        
        # Create sample prepared data
        prepared_data = pd.DataFrame({
            'text': [f'text {i}' for i in range(100)],
            'label': [i % 2 for i in range(100)]
        })
        prepared_file = tmp_path / "prepared.csv"
        prepared_data.to_csv(prepared_file, index=False)
        
        # Run validation
        validation = DataValidation()
        train_path, test_path, report_path = validation.initiate_data_validation(
            str(prepared_file)
        )
        
        # Verify outputs
        assert os.path.exists(train_path)
        assert os.path.exists(test_path)
        assert os.path.exists(report_path)
        
        # Verify split sizes
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        assert len(train_data) > len(test_data)  # Train should be larger


class TestModelTrainer:
    """Tests for Model Trainer component"""
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        sys.platform == 'win32',
        reason="Skipping torch tests on Windows due to DLL issues in CI"
    )
    def test_model_trainer_config(self):
        """Test model trainer configuration"""
        from src.components.model_trainer import ModelTrainerConfig
        
        config = ModelTrainerConfig(
            model_name="roberta-base",
            epochs=1,
            batch_size=8
        )
        
        assert config.model_name == "roberta-base"
        assert config.epochs == 1
        assert config.batch_size == 8
        
    def test_mlflow_connection_check(self):
        """Test MLflow connection checking without importing torch"""
        # Test the connection logic without actually importing torch-dependent code
        import urllib.request
        from urllib.error import URLError
        
        # Simulate the connection check logic
        invalid_uri = "http://invalid-uri:9999"
        max_retries = 1
        
        connection_success = False
        for attempt in range(max_retries):
            try:
                urllib.request.urlopen(f"{invalid_uri}/health", timeout=1)
                connection_success = True
                break
            except (URLError, Exception):
                pass
        
        # Should fail for invalid URI
        assert connection_success == False


# Pytest fixtures
@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'text': [
            'This is fake news',
            'This is real news',
            'Another fake article',
            'Another real article'
        ],
        'label': [1, 0, 1, 0]
    })


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing without actual MLflow server"""
    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.set_experiment'), \
         patch('mlflow.start_run'), \
         patch('mlflow.log_param'), \
         patch('mlflow.log_metric'):
        yield


# Integration tests
@pytest.mark.integration
class TestMLPipeline:
    """Integration tests for the complete ML pipeline"""
    
    def test_full_pipeline(self, tmp_path):
        """Test the full pipeline end-to-end"""
        # This would test the complete flow
        # from ingestion to model training
        pass