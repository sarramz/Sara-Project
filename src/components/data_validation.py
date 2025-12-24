"""
Data Validation Module
Step 3: Validate data quality and split into train/test
"""

import os
import sys
import json
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from src.exceptions import CustomException
from src.logger import logging


@dataclass
class DataValidationConfig:
    """Configuration for data validation"""
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    validation_report_path: str = os.path.join("artifacts", "validation_report.json")
    
    # Validation thresholds
    required_columns: List[str] = None
    min_samples: int = 50  # Reduced from 100 to handle smaller datasets
    max_missing_ratio: float = 0.1  # Max 10% missing values
    min_text_length: int = 10
    valid_labels: List[int] = None
    test_size: float = 0.2
    random_state: int = 42

    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ["text", "label"]
        if self.valid_labels is None:
            self.valid_labels = [0, 1]  # Binary classification


class DataValidation:
    """Validate data quality and split into train/test sets"""

    def __init__(self):
        self.config = DataValidationConfig()
        self.validation_errors = []
        self.validation_warnings = []
        logging.info("Data Validation initialized")

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate dataset schema

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        logging.info("Validating schema...")

        # Check required columns
        missing_cols = set(self.config.required_columns) - set(df.columns)
        if missing_cols:
            error = f"Missing required columns: {missing_cols}"
            self.validation_errors.append(error)
            logging.error(f" ERROR {error}")
            return False

        logging.info(" Schema validation passed")
        return True

    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        logging.info("Validating data quality...")
        is_valid = True

        # Check minimum samples
        if len(df) < self.config.min_samples:
            error = f"Insufficient samples: {len(df)} < {self.config.min_samples}"
            self.validation_errors.append(error)
            logging.error(f" ERROR{error}")
            is_valid = False

        # Check missing values
        for col in self.config.required_columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > self.config.max_missing_ratio:
                error = f"Too many missing values in {col}: {missing_ratio:.2%}"
                self.validation_errors.append(error)
                logging.error(f" ERROR {error}")
                is_valid = False

        # Check text length
        if "text" in df.columns:
            df["text_length"] = df["text"].astype(str).str.len()
            short_texts = (df["text_length"] < self.config.min_text_length).sum()
            if short_texts > len(df) * 0.1:  # More than 10% short texts
                warning = f"Many short texts detected: {short_texts} samples ({short_texts/len(df):.1%})"
                self.validation_warnings.append(warning)
                logging.warning(f" WARNING {warning}")

        # Check labels
        if "label" in df.columns:
            invalid_labels = ~df["label"].isin(self.config.valid_labels)
            if invalid_labels.any():
                error = f"Invalid labels found: {df[invalid_labels]['label'].unique()}"
                self.validation_errors.append(error)
                logging.error(f" ERROR {error}")
                is_valid = False

            # Check label balance
            label_counts = df["label"].value_counts()
            min_label_ratio = label_counts.min() / len(df)
            if min_label_ratio < 0.1:  # Less than 10%
                warning = f"Imbalanced dataset: {label_counts.to_dict()}"
                self.validation_warnings.append(warning)
                logging.warning(f" WARNING  {warning}")

        if is_valid:
            logging.info(" ERROR Data quality validation passed")
        else:
            logging.error("ERROR Data quality validation failed")

        return is_valid

    def validate_labels(self, df: pd.DataFrame) -> bool:
        """
        Validate labels specifically

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        logging.info("Validating labels...")

        if "label" not in df.columns:
            error = "Label column not found"
            self.validation_errors.append(error)
            logging.error(f" ERROR{error}")
            return False

        # Convert labels to int if needed
        try:
            df["label"] = df["label"].astype(int)
        except (ValueError, TypeError) as e:
            error = f"Cannot convert labels to integer: {str(e)}"
            self.validation_errors.append(error)
            logging.error(f" ERROR {error}")
            return False

        # Check label types
        unique_labels = df["label"].unique()
        invalid = set(unique_labels) - set(self.config.valid_labels)
        if invalid:
            error = f"Invalid label values: {invalid}"
            self.validation_errors.append(error)
            logging.error(f" ERROR {error}")
            return False

        # Check label distribution
        label_dist = df["label"].value_counts()
        logging.info(f"Label distribution:\n{label_dist}")

        if len(label_dist) < 2:
            error = "Only one class present in labels"
            self.validation_errors.append(error)
            logging.error(f"ERROR {error}")
            return False

        logging.info("  Label validation passed")
        return True

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets

        Args:
            df: DataFrame to split

        Returns:
            Tuple of (train_df, test_df)
        """
        logging.info(f"Splitting data: test_size={self.config.test_size}")

        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df["label"]
        )

        logging.info(f"Train samples: {len(train_df)}")
        logging.info(f"Test samples: {len(test_df)}")

        # Verify label distribution in splits
        train_dist = train_df["label"].value_counts()
        test_dist = test_df["label"].value_counts()

        logging.info(f"Train label distribution:\n{train_dist}")
        logging.info(f"Test label distribution:\n{test_dist}")

        return train_df, test_df

    def initiate_data_validation(self, prepared_data_path):
        """
        Run all validations and split data

        Args:
            prepared_data_path: Path to prepared data CSV

        Returns:
            Tuple of (train_path, test_path, validation_report_path)
        """
        logging.info("=" * 70)
        logging.info("STEP 3: DATA VALIDATION")
        logging.info("=" * 70)

        try:
            # Reset error/warning lists
            self.validation_errors = []
            self.validation_warnings = []

            # Load prepared data
            logging.info(f"Loading prepared data from {prepared_data_path}")
            df = pd.read_csv(prepared_data_path)
            logging.info(f"Loaded {len(df)} samples")

            # Run validations
            schema_valid = self.validate_schema(df)
            quality_valid = self.validate_data_quality(df)
            labels_valid = self.validate_labels(df)

            is_valid = schema_valid and quality_valid and labels_valid

            # Create validation report
            validation_report = {
                "is_valid": is_valid,
                "validations": {
                    "schema_valid": schema_valid,
                    "quality_valid": quality_valid,
                    "labels_valid": labels_valid,
                },
                "errors": self.validation_errors,
                "warnings": self.validation_warnings,
                "statistics": {
                    "total_samples": len(df),
                    "label_distribution": df["label"].value_counts().to_dict(),
                    "avg_text_length": float(df["text"].str.len().mean()),
                    "min_text_length": int(df["text"].str.len().min()),
                    "max_text_length": int(df["text"].str.len().max()),
                }
            }

            if not is_valid:
                error_msg = f"Validation failed with {len(self.validation_errors)} errors"
                logging.error(f"ERROR {error_msg}")
                logging.error(f"Errors: {self.validation_errors}")
                
                # Print errors to console for DVC visibility
                print("\n" + "=" * 70)
                print("VALIDATION ERRORS:")
                print("=" * 70)
                for i, error in enumerate(self.validation_errors, 1):
                    print(f"{i}. {error}")
                print("=" * 70 + "\n")
                
                # Save validation report with errors
                os.makedirs(os.path.dirname(self.config.validation_report_path), exist_ok=True)
                with open(self.config.validation_report_path, "w") as f:
                    json.dump(validation_report, f, indent=2)
                
                raise ValueError(error_msg)

            logging.info("  All validations passed")

            # Split data into train and test
            train_df, test_df = self.split_data(df)

            # Add split information to report
            validation_report["splits"] = {
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "test_size": self.config.test_size,
                "train_label_distribution": train_df["label"].value_counts().to_dict(),
                "test_label_distribution": test_df["label"].value_counts().to_dict(),
            }

            # Save train and test sets
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            train_df.to_csv(self.config.train_data_path, index=False, header=True)
            test_df.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info(f" Train data saved to: {self.config.train_data_path}")
            logging.info(f" Test data saved to: {self.config.test_data_path}")

            # Save validation report
            with open(self.config.validation_report_path, "w") as f:
                json.dump(validation_report, f, indent=2)

            logging.info(f" Validation report saved to: {self.config.validation_report_path}")

            return (
                self.config.train_data_path,
                self.config.test_data_path,
                self.config.validation_report_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Validation Step")
    parser.add_argument(
        "--prepared-data",
        default="artifacts/prepared_data.csv",
        help="Path to prepared data"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.prepared_data):
        print(f" Prepared data not found: {args.prepared_data}")
        print("Run data preparation first: python -m src.components.data_preparation")
        sys.exit(1)
    
    validation = DataValidation()
    train_path, test_path, report_path = validation.initiate_data_validation(
        args.prepared_data
    )
    
    print(f"\n Data validation completed!")
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")
    print(f"Validation report: {report_path}")