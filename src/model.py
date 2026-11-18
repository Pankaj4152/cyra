"""
CNN Architecture for Malware Detection
======================================
Implements a 3-block Convolutional Neural Network (CNN) tailored for grayscale
binary executable images. Uses GlobalAveragePooling2D and Dropout regularization
to prevent overfitting.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(128, 128, 1), learning_rate=0.001) -> keras.Model:
    """
    Constructs and compiles the 3-block CNN binary classifier.
    
    Architecture:
    - Input: (128, 128, 1)
    - Block 1: Conv2D(32, 3x3, ReLU, same padding) + MaxPooling2D(2x2)
    - Block 2: Conv2D(64, 3x3, ReLU, same padding) + MaxPooling2D(2x2)
    - Block 3: Conv2D(128, 3x3, ReLU, same padding) + MaxPooling2D(2x2)
    - Global Pooling: GlobalAveragePooling2D()
    - Fully Connected: Dense(64, ReLU) + Dropout(0.3)
    - Output: Dense(1, Sigmoid) for binary probability (0 = Benign, 1 = Malicious)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"),
        layers.MaxPooling2D((2, 2), name="pool3"),
        
        # Classification Head
        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dense(64, activation="relu", name="dense1"),
        layers.Dropout(0.3, name="dropout"),
        layers.Dense(1, activation="sigmoid", name="output")
    ], name="MalwareVision_CNN")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
