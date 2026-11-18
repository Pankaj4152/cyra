"""
Image Preprocessing Pipeline
============================
Handles standard 128x128 grayscale conversion, float32 normalization to [0.0, 1.0],
and single-channel dimension expansion for CNN input.
Used consistently across dataset loading, training, evaluation, and live inference.
"""

import os
from pathlib import Path
from typing import Tuple, List, Union
import numpy as np
from PIL import Image

IMAGE_SIZE: Tuple[int, int] = (128, 128)
INPUT_SHAPE: Tuple[int, int, int] = (128, 128, 1)


def preprocess_image(image: Union[Image.Image, str, Path]) -> np.ndarray:
    """
    Preprocess an input image for the CNN model.
    
    Steps:
    1. Open if path is provided.
    2. Convert to 8-bit grayscale ('L').
    3. Resize to 128x128 pixels using bilinear resampling.
    4. Convert to float32 NumPy array.
    5. Normalize pixel intensity values from [0, 255] to [0.0, 1.0].
    6. Expand dimensions to shape (128, 128, 1).
    
    Returns:
        np.ndarray with shape (128, 128, 1) and dtype float32.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    
    # Ensure grayscale
    image = image.convert("L")
    
    # Resize to standard CNN input dimension
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    
    # Convert to float32 and normalize
    image_arr = np.asarray(image, dtype=np.float32) / 255.0
    
    # Add channel dimension: (128, 128, 1)
    return np.expand_dims(image_arr, axis=-1)


def load_dataset_split(base_dir: Union[str, Path], split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads all benign (0) and malicious (1) images from a specific split directory
    (e.g., dataset/train, dataset/val, dataset/test).
    
    Returns:
        X: numpy array of shape (N, 128, 128, 1)
        y: numpy array of shape (N,)
    """
    split_dir = Path(base_dir) / split
    benign_dir = split_dir / "benign"
    malicious_dir = split_dir / "malicious"
    
    images: List[np.ndarray] = []
    labels: List[int] = []
    
    # Benign files (label = 0)
    for fname in os.listdir(benign_dir):
        fpath = benign_dir / fname
        if fpath.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            try:
                images.append(preprocess_image(fpath))
                labels.append(0)
            except Exception as e:
                print(f"Warning: skipping corrupted file {fpath}: {e}")
                
    # Malicious files (label = 1)
    for fname in os.listdir(malicious_dir):
        fpath = malicious_dir / fname
        if fpath.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            try:
                images.append(preprocess_image(fpath))
                labels.append(1)
            except Exception as e:
                print(f"Warning: skipping corrupted file {fpath}: {e}")
                
    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    
    return X, y
