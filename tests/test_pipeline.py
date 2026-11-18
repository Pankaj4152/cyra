"""
Comprehensive Verification & Test Suite
========================================
Validates all requirements of the MalwareVision project:
1. Preprocessing (Grayscale conversion, resize to 128x128, float32 normalization)
2. Prediction pipeline (Benign vs Malicious classification with confidence scores)
3. Robustness on arbitrary dimensions and RGB images
4. Graceful handling of invalid files
5. Existence and validity of metrics.json and all exported plots
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_image, IMAGE_SIZE
from src.predict import predict, get_model


def run_tests():
    print("=" * 60)
    print("RUNNING AUTOMATED TEST SUITE FOR MALWAREVISION")
    print("=" * 60)
    passed = 0
    total = 0

    def test(name, condition):
        nonlocal passed, total
        total += 1
        status = "PASSED ✅" if condition else "FAILED ❌"
        print(f"[{total}] {name}: {status}")
        if condition:
            passed += 1

    # Test 1: Preprocessing shape and normalization
    dummy_rgb = Image.new("RGB", (256, 512), color=(120, 200, 50))
    proc = preprocess_image(dummy_rgb)
    test("Preprocessing converts RGB (256x512) to (128, 128, 1)", proc.shape == (128, 128, 1))
    test("Preprocessing outputs float32 in range [0.0, 1.0]", proc.dtype == np.float32 and proc.min() >= 0.0 and proc.max() <= 1.0)

    # Test 2: Model file existence and loading
    model = get_model()
    test("Trained model loads successfully with expected input shape", model.input_shape == (None, 128, 128, 1))

    # Test 3: Inference on Benign demo sample
    benign_samples = os.listdir("samples/benign")
    b_sample_path = os.path.join("samples/benign", benign_samples[0])
    b_res = predict(b_sample_path)
    test(f"Inference runs on benign sample ({benign_samples[0][:20]}...)", b_res["prediction"] in ["Benign", "Malicious"])
    test("Inference returns confidence and probability between 0 and 1", 0.0 <= b_res["confidence"] <= 1.0 and 0.0 <= b_res["malware_probability"] <= 1.0)

    # Test 4: Inference on Malicious demo sample
    mal_samples = os.listdir("samples/malicious")
    m_sample_path = os.path.join("samples/malicious", mal_samples[0])
    m_res = predict(m_sample_path)
    test(f"Inference runs on malicious sample ({mal_samples[0][:20]}...)", m_res["prediction"] in ["Benign", "Malicious"])

    # Test 5: Metrics file validity
    metrics_path = "results/metrics.json"
    has_metrics = os.path.exists(metrics_path)
    test("results/metrics.json exists", has_metrics)
    if has_metrics:
        with open(metrics_path, "r") as f:
            m_data = json.load(f)
        test("Accuracy is >= 90%", m_data.get("accuracy", 0) >= 0.90)
        test("Precision is >= 85%", m_data.get("precision", 0) >= 0.85)
        test("Recall is >= 85%", m_data.get("recall", 0) >= 0.85)
        test("F1-score is >= 85%", m_data.get("f1_score", 0) >= 0.85)

    # Test 6: Visualizations exist and are non-empty
    for plot_name in ["accuracy_graph.png", "loss_graph.png", "confusion_matrix.png"]:
        p_path = os.path.join("results", plot_name)
        test(f"Plot '{plot_name}' exists and has content", os.path.exists(p_path) and os.path.getsize(p_path) > 1000)

    # Test 7: Training history JSON exists
    hist_path = "results/training_history.json"
    test("results/training_history.json exists", os.path.exists(hist_path) and os.path.getsize(hist_path) > 100)

    # Test 8: Graceful error handling on non-existent file
    error_caught = False
    try:
        predict("non_existent_file_xyz.png")
    except Exception:
        error_caught = True
    test("Graceful error raised when non-existent image is passed", error_caught)

    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
