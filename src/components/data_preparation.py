"""
Data Preparation Module
Step 2: Clean and prepare text data
"""

import os
import sys
import re
import pandas as pd
from dataclasses import dataclass
from src.exceptions import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
import argparse
@dataclass
class DataPreparationConfig:
    """Configuration for data preparation"""
    prepared_data_path: str = os.path.join("artifacts", "prepared_data.csv")
    min_text_length: int = 10
    max_text_length: int = 512


class DataPreparation:
    """Clean and prepare text data for model training"""

    def __init__(self):
        self.config = DataPreparationConfig()
        logging.info("Data Preparation initialized")

    def clean_text(self, text):
        """
        Clean text data

        Args:
            text: Raw text string

        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"www\S+", "", text)  # Remove www links
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # Remove special chars
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces

        return text

    def combine_title_text(self, title, text):
        """
        Combine title and text with separator

        Args:
            title: Article title
            text: Article text

        Returns:
            Combined string with [SEP] token
        """
        title_clean = self.clean_text(title) if title else ""
        text_clean = self.clean_text(text) if text else ""

        if title_clean and text_clean:
            return f"{title_clean} [SEP] {text_clean}"
        elif text_clean:
            return text_clean
        elif title_clean:
            return title_clean
        else:
            return ""

    def initiate_data_preparation(self, raw_data_path):
        """
        Prepare dataset for training

        Args:
            raw_data_path: Path to raw data CSV

        Returns:
            Path to prepared data
        """
        logging.info("=" * 70)
        logging.info("STEP 2: DATA PREPARATION")
        logging.info("=" * 70)

        try:
            # Load raw data
            logging.info(f"Loading raw data from {raw_data_path}")
            df = pd.read_csv(raw_data_path)
            initial_count = len(df)
            logging.info(f"Loaded {initial_count} samples")

            # Clean text
            logging.info("Cleaning text data...")
            df["text_clean"] = df["text"].apply(self.clean_text)
            df["title_clean"] = df["title"].apply(self.clean_text)

            # Combine title and text
            logging.info("Combining title and text...")
            df["combined"] = df.apply(
                lambda row: self.combine_title_text(row["title"], row["text"]), 
                axis=1
            )

            # Filter out short texts
            logging.info(f"Filtering texts shorter than {self.config.min_text_length} characters...")
            df = df[df["combined"].str.len() >= self.config.min_text_length]
            after_short_filter = len(df)
            logging.info(f"Removed {initial_count - after_short_filter} short texts")

            # Filter out very long texts
            logging.info(f"Filtering texts longer than {self.config.max_text_length} characters...")
            df = df[df["combined"].str.len() <= self.config.max_text_length]
            after_long_filter = len(df)
            logging.info(f"Removed {after_short_filter - after_long_filter} very long texts")

            # Drop rows with NaN labels
            df = df.dropna(subset=["label"])
            after_nan_filter = len(df)
            logging.info(f"Removed {after_long_filter - after_nan_filter} rows with missing labels")

            # Keep only necessary columns
            df_final = df[["combined", "label"]].copy()
            df_final.columns = ["text", "label"]

            # Get statistics
            stats = {
                "total_samples": len(df_final),
                "label_distribution": df_final["label"].value_counts().to_dict(),
                "avg_text_length": df_final["text"].str.len().mean(),
                "min_text_length": df_final["text"].str.len().min(),
                "max_text_length": df_final["text"].str.len().max(),
            }

            logging.info("Data Statistics:")
            for key, value in stats.items():
                logging.info(f"  {key}: {value}")

            # Save prepared data
            os.makedirs(os.path.dirname(self.config.prepared_data_path), exist_ok=True)
            df_final.to_csv(self.config.prepared_data_path, index=False, header=True)

            logging.info(f" Prepared data saved to: {self.config.prepared_data_path}")
            logging.info(f"Final sample count: {len(df_final)}")

            return self.config.prepared_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test data preparation

    parser = argparse.ArgumentParser(description="Data Preparation Step")
    parser.add_argument(
        "--raw-data", 
        default="artifacts/raw_data.csv", 
        help="Path to raw data"
    )
    args = parser.parse_args()
    # First run ingestion
    ingestion = DataIngestion()
    raw_data_path = ingestion.initiate_data_ingestion(
        data_path="data/fake_news.csv",
        sample_size=1000
    )

    # Then run preparation
    preparation = DataPreparation()
    prepared_data_path = preparation.initiate_data_preparation(raw_data_path)

    print(f"\n Data preparation completed!")
    print(f"Prepared data saved to: {prepared_data_path}")