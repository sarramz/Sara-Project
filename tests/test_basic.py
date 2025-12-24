"""
Basic unit tests for CI/CD pipeline
"""
import pytest
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestProjectStructure:
    """Test that project structure is correct"""
    
    def test_src_directory_exists(self):
        """Test that src directory exists"""
        assert os.path.exists('src')
        
    def test_components_directory_exists(self):
        """Test that components directory exists"""
        assert os.path.exists('src/components')
        
    def test_logger_module_exists(self):
        """Test that logger module exists"""
        assert os.path.exists('src/logger.py')
        
    def test_exceptions_module_exists(self):
        """Test that exceptions module exists"""
        assert os.path.exists('src/exceptions.py')


class TestDataIngestion:
    """Tests for Data Ingestion component"""
    
    def test_data_ingestion_imports(self):
        """Test that DataIngestion can be imported"""
        from src.components.data_ingestion import DataIngestion
        assert DataIngestion is not None
        
    def test_data_ingestion_creates_instance(self):
        """Test that DataIngestion instance can be created"""
        from src.components.data_ingestion import DataIngestion
        ingestion = DataIngestion()
        assert ingestion is not None
        
    def test_data_ingestion_with_sample_data(self, tmp_path):
        """Test data ingestion with small sample"""
        from src.components.data_ingestion import DataIngestion
        
        # Create minimal sample data
        sample_data = pd.DataFrame({
            'title': ['Title 1', 'Title 2', 'Title 3'],
            'text': ['Text 1', 'Text 2', 'Text 3'],
            'label': [0, 1, 0]
        })
        sample_file = tmp_path / "test_data.csv"
        sample_data.to_csv(sample_file, index=False)
        
        # Run ingestion with very small sample
        ingestion = DataIngestion()
        output_path = ingestion.initiate_data_ingestion(
            data_path=str(sample_file),
            sample_size=3
        )
        assert os.path.exists(output_path)


class TestDataPreparation:
    """Tests for Data Preparation component"""
    
    def test_data_preparation_imports(self):
        """Test that DataPreparation can be imported"""
        from src.components.data_preparation import DataPreparation
        assert DataPreparation is not None
        
    def test_data_preparation_creates_instance(self):
        """Test that DataPreparation instance can be created"""
        from src.components.data_preparation import DataPreparation
        preparation = DataPreparation()
        assert preparation is not None
        
    def test_data_preparation_with_valid_data(self, tmp_path):
        """Test data preparation with properly formatted data"""
        from src.components.data_preparation import DataPreparation
        
        # Create data with all required columns
        raw_data = pd.DataFrame({
            'title': ['Breaking News', 'Important Update', 'Latest Report'] * 10,
            'text': ['This is some fake news content'] * 30,
            'label': [1, 0, 1] * 10
        })
        raw_file = tmp_path / "raw.csv"
        raw_data.to_csv(raw_file, index=False)
        
        # Run preparation
        preparation = DataPreparation()
        output_path = preparation.initiate_data_preparation(str(raw_file))
        
        # Verify output
        assert os.path.exists(output_path)
        cleaned_data = pd.read_csv(output_path)
        assert len(cleaned_data) > 0
        assert 'text' in cleaned_data.columns
        assert 'label' in cleaned_data.columns


class TestDataValidation:
    """Tests for Data Validation component"""
    
    def test_data_validation_imports(self):
        """Test that DataValidation can be imported"""
        from src.components.data_validation import DataValidation
        assert DataValidation is not None
        
    def test_data_validation_creates_instance(self):
        """Test that DataValidation instance can be created"""
        from src.components.data_validation import DataValidation
        validation = DataValidation()
        assert validation is not None


class TestUtilities:
    """Test utility functions"""
    
    def test_logger_imports(self):
        """Test that logger can be imported"""
        from src.logger import logging
        assert logging is not None
        
    def test_exceptions_imports(self):
        """Test that CustomException can be imported"""
        from src.exceptions import CustomException
        assert CustomException is not None


class TestDataFrameOperations:
    """Test basic DataFrame operations used in the project"""
    
    def test_dataframe_creation(self):
        """Test basic DataFrame creation"""
        df = pd.DataFrame({
            'text': ['sample text'],
            'label': [0]
        })
        assert len(df) == 1
        
    def test_dataframe_filtering(self):
        """Test DataFrame filtering"""
        df = pd.DataFrame({
            'text': ['text1', 'text2', 'text3'],
            'label': [0, 1, 0]
        })
        filtered = df[df['label'] == 0]
        assert len(filtered) == 2
        
    def test_dataframe_column_operations(self):
        """Test DataFrame column operations"""
        df = pd.DataFrame({
            'text': ['UPPER', 'lower', 'MiXeD'],
        })
        df['text_lower'] = df['text'].str.lower()
        assert df['text_lower'].iloc[0] == 'upper'