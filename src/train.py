"""
Model Training Pipeline
=======================
Loads the prepared training and validation datasets, trains the CNN model
with EarlyStopping and ModelCheckpoint callbacks, saves the best model checkpoint to
'models/malware_cnn.keras', and exports actual training loss & accuracy curves to 'results/'.
"""

import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.preprocessing import load_dataset_split
from src.model import build_model


def train_pipeline(
    data_dir: str = "dataset",
    model_save_path: str = "models/malware_cnn.keras",
    results_dir: str = "results",
    epochs: int = 20,
    batch_size: int = 32
):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Step 1: Loading Training and Validation Datasets...")
    print("=" * 60)
    X_train, y_train = load_dataset_split(data_dir, split="train")
    X_val, y_val = load_dataset_split(data_dir, split="val")

    print(f"Training samples:   {X_train.shape[0]} (Benign: {sum(y_train == 0)}, Malicious: {sum(y_train == 1)})")
    print(f"Validation samples: {X_val.shape[0]} (Benign: {sum(y_val == 0)}, Malicious: {sum(y_val == 1)})")
    print(f"Input shape: {X_train.shape[1:]}")

    print("\n" + "=" * 60)
    print("Step 2: Building and Compiling CNN Model...")
    print("=" * 60)
    model = build_model(input_shape=X_train.shape[1:])

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n" + "=" * 60)
    print(f"Step 3: Training Model (up to {epochs} epochs, batch size {batch_size})...")
    print("=" * 60)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nModel training finished. Best checkpoint saved at: {model_save_path}")

    # Step 4: Export Graphs and History
    print("\n" + "=" * 60)
    print("Step 4: Generating Training Visualizations & Saving History...")
    print("=" * 60)
    
    epochs_ran = len(history.history["loss"])
    epochs_range = range(1, epochs_ran + 1)

    # 1. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history["accuracy"], label="Training Accuracy", color="#2563EB", marker="o", linewidth=2)
    plt.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", color="#16A34A", marker="s", linewidth=2)
    plt.title("Model Training & Validation Accuracy", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    acc_path = os.path.join(results_dir, "accuracy_graph.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved accuracy graph: {acc_path}")

    # 2. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history["loss"], label="Training Loss", color="#DC2626", marker="o", linewidth=2)
    plt.plot(epochs_range, history.history["val_loss"], label="Validation Loss", color="#EA580C", marker="s", linewidth=2)
    plt.title("Model Training & Validation Loss", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Binary Crossentropy Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    loss_path = os.path.join(results_dir, "loss_graph.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved loss graph: {loss_path}")

    # 3. Save History to JSON
    hist_dict = {
        "epochs": list(epochs_range),
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "stopped_epoch": callbacks[0].stopped_epoch if hasattr(callbacks[0], "stopped_epoch") else epochs_ran
    }
    hist_path = os.path.join(results_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(hist_dict, f, indent=4)
    print(f"Saved training history to: {hist_path}")

    return model, history


if __name__ == "__main__":
    train_pipeline()
