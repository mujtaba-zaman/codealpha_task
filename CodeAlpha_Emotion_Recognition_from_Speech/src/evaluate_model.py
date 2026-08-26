import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import load_model

# -----------------------------
# Load data
# -----------------------------

X = np.load("dataset/processed/X.npy")
y = np.load("dataset/processed/y.npy")
labels = np.load("dataset/processed/labels.npy")

X = X.astype("float32")
X = (X - X.mean()) / (X.std() + 1e-8)

# Same split used during training
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

# -----------------------------
# Load trained model
# -----------------------------

model = load_model("models/emotion_cnn.keras")

# -----------------------------
# Predictions
# -----------------------------

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

# -----------------------------
# Classification Report
# -----------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=labels,
        digits=4
    )
)

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.title("Speech Emotion Recognition - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved to:")
print("models/confusion_matrix.png")