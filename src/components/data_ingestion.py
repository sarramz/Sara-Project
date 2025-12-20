"""
Data Ingestion for Fake News Detection
Step 1: Load raw data and save it
"""

import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exceptions import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion paths"""
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")


class DataIngestion:
    """Load raw data from source"""

    def __init__(self):
        self.config = DataIngestionConfig()
        logging.info("Data Ingestion initialized")

    def initiate_data_ingestion(self, data_path="data/fake_news.csv", sample_size=None):
        """
        Load raw dataset from CSV file

        Args:
            data_path: Path to CSV file with columns [title, text, label]
            sample_size: Optional - limit dataset size for quick testing

        Returns:
            Path to saved raw data
        """
        logging.info("=" * 70)
        logging.info("STEP 1: DATA INGESTION")
        logging.info("=" * 70)

        try:
            # Load dataset
            logging.info(f"Reading dataset from {data_path}")
            df = pd.read_csv(data_path)

            # Verify required columns exist
            required_cols = ["text", "label"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in dataset")

            # Add title column if not present
            if "title" not in df.columns:
                logging.info("'title' column not found, adding empty title column")
                df["title"] = ""

            logging.info(f"Dataset loaded: {df.shape}")
            logging.info(f"Columns: {list(df.columns)}")
            logging.info(f"Label distribution:\n{df['label'].value_counts()}")

            # Sample if requested (for quick testing)
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                logging.info(f"Sampled {sample_size} records for testing")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info(f" Raw data saved to: {self.config.raw_data_path}")
            logging.info(f"Total samples: {len(df)}")

            return self.config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Ingestion Step")
    parser.add_argument(
        "--data",
        default="data/fake_news.csv",
        help="Path to raw data CSV file"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for testing (None = full dataset)"
    )
    args = parser.parse_args()
    
    ingestion = DataIngestion()
    raw_data_path = ingestion.initiate_data_ingestion(
        data_path=args.data,
        sample_size=args.sample_size
    )

    print(f"\n Data ingestion completed!")
    print(f"Raw data saved to: {raw_data_path}")