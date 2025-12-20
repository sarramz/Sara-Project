import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.logger import logging
from src.exceptions import CustomException


class TestUtils:
    """Test cases for utility functions"""

    def test_logger_exists(self):
        """Test if logger is properly configured"""
        assert logging is not None
        # Test logging
        try:
            logging.info("Test log message")
            assert True
        except Exception as e:
            pytest.fail(f"Logger test failed: {str(e)}")

    def test_custom_exception(self):
        """Test custom exception"""
        try:
            raise CustomException("Test error", sys)
        except CustomException as e:
            assert "Test error" in str(e)
        except Exception:
            pytest.fail("CustomException not raised properly")

    def test_artifacts_directory(self):
        """Test if artifacts directory exists"""
        artifacts_dir = "artifacts"
        if not os.path.exists(artifacts_dir):
            pytest.skip("Artifacts directory not found - this is expected for fresh setup")
        else:
            assert os.path.isdir(artifacts_dir)
