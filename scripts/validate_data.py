#!/usr/bin/env python3
"""
Data Validation Script
Validates that the fake_news.csv dataset meets requirements
"""

import os
import sys
import pandas as pd


def validate_dataset(data_path="data/fake_news.csv"):
    """Validate the fake news dataset"""
    print("🔍 Validating dataset...")
    print(f"📁 Dataset path: {data_path}")
    print("=" * 70)

    # Check if file exists
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Dataset not found at {data_path}")
        print("\n💡 Next steps:")
        print("   1. Download a fake news dataset from Kaggle")
        print("   2. Place it as: data/fake_news.csv")
        print("   3. See data/README.md for more information")
        return False

    print(f"✅ File exists: {data_path}")

    # Load dataset
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading dataset: {str(e)}")
        return False

    # Check shape
    print(f"\n📊 Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Check required columns
    print(f"\n📋 Checking required columns...")
    required_cols = ["text", "label"]
    optional_cols = ["title"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Found columns: {list(df.columns)}")
        return False

    print(f"✅ All required columns present: {required_cols}")

    for col in optional_cols:
        if col in df.columns:
            print(f"✅ Optional column found: {col}")
        else:
            print(f"⚠️  Optional column missing: {col} (will use empty strings)")

    # Check data types and nulls
    print(f"\n🔍 Checking data quality...")

    text_col = "text"
    label_col = "label"

    # Text column
    null_text = df[text_col].isnull().sum()
    if null_text > 0:
        print(f"⚠️  Found {null_text} null values in '{text_col}' column ({null_text/len(df)*100:.1f}%)")
    else:
        print(f"✅ No null values in '{text_col}' column")

    # Label column
    null_labels = df[label_col].isnull().sum()
    if null_labels > 0:
        print(f"❌ Found {null_labels} null values in '{label_col}' column")
        return False
    else:
        print(f"✅ No null values in '{label_col}' column")

    # Check label values
    unique_labels = df[label_col].unique()
    print(f"\n📊 Label distribution:")
    print(f"   Unique labels: {sorted(unique_labels)}")

    if not all(label in [0, 1] for label in unique_labels):
        print(f"❌ Labels must be binary (0 or 1)")
        return False

    label_counts = df[label_col].value_counts()
    for label, count in label_counts.items():
        percentage = count / len(df) * 100
        label_name = "Real News" if label == 0 else "Fake News"
        print(f"   {label} ({label_name}): {count:,} ({percentage:.1f}%)")

    # Check dataset size
    min_samples = 100
    if len(df) < min_samples:
        print(f"\n⚠️  Dataset has only {len(df)} samples (recommended: >{min_samples:,})")
    else:
        print(f"\n✅ Dataset size is sufficient: {len(df):,} samples")

    # Check text lengths
    df["text_length"] = df[text_col].fillna("").astype(str).str.len()
    avg_length = df["text_length"].mean()
    min_length = df["text_length"].min()
    max_length = df["text_length"].max()

    print(f"\n📏 Text length statistics:")
    print(f"   Average: {avg_length:.0f} characters")
    print(f"   Minimum: {min_length} characters")
    print(f"   Maximum: {max_length:,} characters")

    if min_length < 10:
        very_short = (df["text_length"] < 10).sum()
        print(f"⚠️  {very_short} texts are very short (<10 characters)")

    # Final summary
    print("\n" + "=" * 70)
    print("✅ Dataset validation PASSED!")
    print("\n💡 Next steps:")
    print("   1. Run the ML pipeline: python run_pipeline.py")
    print("   2. Or use Prefect: bash scripts/run_prefect_flow_docker.sh")
    print("   3. Or use DVC: dvc repro")

    return True


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/fake_news.csv"
    success = validate_dataset(data_path)
    sys.exit(0 if success else 1)
