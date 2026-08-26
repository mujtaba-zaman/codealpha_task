import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# Load processed data
# -----------------------------

X = np.load("dataset/processed/X.npy")
y = np.load("dataset/processed/y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# Normalize MFCC values
X = X.astype("float32")
X = (X - X.mean()) / (X.std() + 1e-8)

# -----------------------------
# Train / Validation / Test
# -----------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# -----------------------------
# CNN Model
# -----------------------------

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(40, 174, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.30),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.40),

    Dense(8, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Save best model
# -----------------------------

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/emotion_cnn.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

# -----------------------------
# Train
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# -----------------------------
# Test
# -----------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nFinal Test Accuracy:", test_accuracy)
print("Final Test Loss:", test_loss)

print("\nTraining completed!")