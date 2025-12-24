"""
Prediction Pipeline for Fake News Detection
Uses trained RoBERTa model for inference
"""

import os
import sys
import re
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import json

from src.exceptions import CustomException
from src.logger import logging


class FakeNewsPredictionPipeline:
    def __init__(self, model_path="artifacts/roberta_fakenews"):
        """
        Initialize prediction pipeline

        Args:
            model_path: Path to trained model directory
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.label_map = None
        self.metadata = None

        self.load_model()

    def clean_text(self, text):
        """Clean text data"""
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_model(self):
        """Load trained model and tokenizer"""
        try:
            logging.info(f"Loading model from {self.model_path}")

            # Load tokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)

            # Load model
            self.model = RobertaForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()  # Set to evaluation mode

            # Load metadata
            metadata_path = os.path.join(self.model_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)

            # Load label map
            label_map_path = os.path.join(self.model_path, "label_map.json")
            if os.path.exists(label_map_path):
                with open(label_map_path, "r") as f:
                    self.label_map = json.load(f)
            else:
                self.label_map = {"0": "Real News", "1": "Fake News"}

            logging.info("✓ Model loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def predict_single(self, title, text):
        """
        Predict whether a single news article is fake or real

        Args:
            title: Article title
            text: Article text

        Returns:
            dict with prediction, confidence, and probabilities
        """
        try:
            # Clean and combine
            title_clean = self.clean_text(title)
            text_clean = self.clean_text(text)
            combined = f"{title_clean} [SEP] {text_clean}"

            # Tokenize
            max_length = self.metadata.get("max_length", 128) if self.metadata else 128
            inputs = self.tokenizer(combined, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()

            # Get label
            prediction_label = self.label_map.get(str(predicted_class), f"Class {predicted_class}")

            result = {
                "prediction": prediction_label,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": {"Real News": probabilities[0][0].item(), "Fake News": probabilities[0][1].item()},
            }

            return result

        except Exception as e:
            raise CustomException(e, sys)

    def predict_batch(self, articles):
        """
        Predict for multiple articles

        Args:
            articles: List of dicts with 'title' and 'text' keys

        Returns:
            List of prediction dicts
        """
        try:
            results = []
            for article in articles:
                result = self.predict_single(title=article.get("title", ""), text=article.get("text", ""))
                results.append(result)

            return results

        except Exception as e:
            raise CustomException(e, sys)


# Flask-compatible custom data class
class CustomData:
    def __init__(self, title: str, text: str):
        self.title = title
        self.text = text

    def to_dict(self):
        return {"title": self.title, "text": self.text}


if __name__ == "__main__":
    # Test the prediction pipeline
    print("Testing Fake News Prediction Pipeline...")

    try:
        # Initialize pipeline
        pipeline = FakeNewsPredictionPipeline()

        # Test examples
        test_articles = [
            {
                "title": "Scientists Discover New Treatment for Cancer",
                "text": "Researchers at a major university have made a breakthrough in cancer treatment that could save millions of lives.",
            },
            {
                "title": "BREAKING: Aliens Land in New York City!!!",
                "text": "Unbelievable footage shows alien spacecraft landing in Times Square. Government officials are hiding the truth!",
            },
        ]

        # Predict
        print("\n" + "=" * 70)
        for i, article in enumerate(test_articles, 1):
            print(f"\n Article {i}:")
            print(f"Title: {article['title']}")
            print(f"Text: {article['text'][:100]}...")

            result = pipeline.predict_single(article["title"], article["text"])

            print(f"\n Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print("   Probabilities:")
            print(f"     - Real News: {result['probabilities']['Real News']:.2%}")
            print(f"     - Fake News: {result['probabilities']['Fake News']:.2%}")
            print("=" * 70)

    except Exception as e:
        print(f" Error: {str(e)}")
        print("Make sure you have a trained model in artifacts/roberta_fakenews/")
