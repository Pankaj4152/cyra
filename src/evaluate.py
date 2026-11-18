"""
Model Evaluation Pipeline
=========================
Evaluates the trained CNN on the held-out unseen test set (600 samples).
Calculates Accuracy, Precision, Recall, and F1-Score.
Generates an annotated Confusion Matrix heatmap and exports metrics to 'results/metrics.json'.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from src.preprocessing import load_dataset_split


def evaluate_pipeline(
    model_path: str = "models/malware_cnn.keras",
    data_dir: str = "dataset",
    results_dir: str = "results",
    threshold: float = 0.5
):
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Step 1: Loading Unseen Test Dataset...")
    print("=" * 60)
    X_test, y_test = load_dataset_split(data_dir, split="test")
    print(f"Test samples loaded: {X_test.shape[0]} (Benign: {sum(y_test == 0)}, Malicious: {sum(y_test == 1)})")

    print("\n" + "=" * 60)
    print(f"Step 2: Loading Trained Model from {model_path}...")
    print("=" * 60)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")

    model = load_model(model_path)

    print("\n" + "=" * 60)
    print("Step 3: Generating Model Predictions on Test Split...")
    print("=" * 60)
    probabilities = model.predict(X_test, batch_size=32, verbose=1)
    predictions = (probabilities.ravel() >= threshold).astype(int)

    # Calculate metrics
    acc = float(accuracy_score(y_test, predictions))
    prec = float(precision_score(y_test, predictions, zero_division=0))
    rec = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))
    cm = confusion_matrix(y_test, predictions)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "threshold": threshold,
        "test_samples": int(len(y_test)),
        "benign_samples": int(sum(y_test == 0)),
        "malicious_samples": int(sum(y_test == 1)),
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1])
        }
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (UNSEEN TEST DATA):")
    print("=" * 60)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["Benign", "Malicious"], digits=4))

    # Save metrics JSON
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to: {metrics_path}")

    # Step 4: Render Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign (0)", "Malicious (1)"],
        yticklabels=["Benign (0)", "Malicious (1)"],
        annot_kws={"size": 14, "fontweight": "bold"},
        cbar=True
    )
    plt.title(f"Test Confusion Matrix (Acc: {acc * 100:.2f}%, F1: {f1 * 100:.2f}%)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Actual True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix visualization: {cm_path}")

    return metrics


if __name__ == "__main__":
    evaluate_pipeline()
