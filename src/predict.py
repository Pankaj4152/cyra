"""
Inference & Prediction Pipeline
===============================
Provides the inference function for the MalwareVision application.
Preprocesses an input grayscale sample image, executes model forward-pass,
and returns structured classification label, confidence score, and raw probability.
"""

import os
from pathlib import Path
from typing import Dict, Union
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_image

MODEL_PATH = "models/malware_cnn.keras"
_cached_model = None


def get_model(model_path: str = MODEL_PATH):
    """Loads and caches the trained CNN model checkpoint."""
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found! Please train the model using 'python src/train.py' first."
            )
        _cached_model = load_model(model_path)
    return _cached_model


def predict(
    image: Union[Image.Image, str, Path],
    model_path: str = MODEL_PATH,
    threshold: float = 0.5
) -> Dict[str, Union[str, float]]:
    """
    Classifies a sample binary image as Benign or Malicious.
    
    Args:
        image: PIL Image instance or path to an image file.
        model_path: Path to the trained .keras model file.
        threshold: Classification decision boundary (default 0.5).
        
    Returns:
        dict: {
            "prediction": "Malicious" | "Benign",
            "confidence": float (0.0 to 1.0),
            "malware_probability": float (0.0 to 1.0)
        }
    """
    model = get_model(model_path)
    
    # Preprocess to (128, 128, 1)
    processed_img = preprocess_image(image)
    
    # Expand to batch dimension: (1, 128, 128, 1)
    input_batch = np.expand_dims(processed_img, axis=0)
    
    # Model inference
    raw_pred = model.predict(input_batch, verbose=0)
    malware_prob = float(raw_pred[0][0])
    
    is_malicious = malware_prob >= threshold
    confidence = malware_prob if is_malicious else (1.0 - malware_prob)
    
    return {
        "prediction": "Malicious" if is_malicious else "Benign",
        "confidence": confidence,
        "malware_probability": malware_prob
    }


if __name__ == "__main__":
    # Test on a demo sample if model exists
    import sys
    test_sample = "samples/malicious"
    if os.path.exists(test_sample) and os.listdir(test_sample):
        sample_path = os.path.join(test_sample, os.listdir(test_sample)[0])
        print(f"Testing prediction on: {sample_path}")
        try:
            res = predict(sample_path)
            print("Prediction output:", res)
        except Exception as e:
            print("Notice:", e)
