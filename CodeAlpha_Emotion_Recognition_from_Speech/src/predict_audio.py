import os
import sys
import numpy as np
import librosa
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/emotion_cnn.keras"
LABELS_PATH = "dataset/processed/labels.npy"

SAMPLE_RATE = 22050
N_MFCC = 40
MFCC_WIDTH = 174


# ============================================================
# LOAD MODEL AND LABELS
# ============================================================

def load_emotion_model():
    """
    Load the trained CNN model and emotion labels.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Please train the model first using train.py."
        )

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_PATH}\n"
            "Please run prepare_data.py first."
        )

    model = load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH)

    return model, labels


# ============================================================
# AUDIO FEATURE EXTRACTION
# ============================================================

def extract_mfcc(file_path):
    """
    Load an audio file and extract MFCC features
    using the same preprocessing used during training.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Audio file not found: {file_path}"
        )

    # Load audio
    audio, sample_rate = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC
    )

    # Make MFCC fixed size
    mfcc = librosa.util.fix_length(
        mfcc,
        size=MFCC_WIDTH,
        axis=1
    )

    # Convert to float32
    mfcc = mfcc.astype("float32")

    # Normalize using the same method as training
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    # Add channel dimension
    # (40, 174) → (40, 174, 1)
    mfcc = mfcc[..., np.newaxis]

    # Add batch dimension
    # (40, 174, 1) → (1, 40, 174, 1)
    mfcc = np.expand_dims(mfcc, axis=0)

    return mfcc


# ============================================================
# EMOTION PREDICTION
# ============================================================

def predict_emotion(file_path):
    """
    Predict the emotion of an audio file.

    Returns:
        predicted_emotion
        confidence
        probabilities
    """

    # Load model and labels
    model, labels = load_emotion_model()

    # Extract MFCC
    features = extract_mfcc(file_path)

    # Model prediction
    predictions = model.predict(
        features,
        verbose=0
    )

    # Get predicted class
    predicted_index = np.argmax(predictions[0])

    predicted_emotion = labels[predicted_index]

    # Get confidence
    confidence = float(
        predictions[0][predicted_index]
    )

    # Convert probabilities into dictionary
    probabilities = {}

    for i, label in enumerate(labels):
        probabilities[str(label)] = float(
            predictions[0][i]
        )

    return (
        str(predicted_emotion),
        confidence,
        probabilities
    )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

def display_prediction(
    file_path,
    predicted_emotion,
    confidence,
    probabilities
):
    """
    Display prediction results in terminal.
    """

    print("\n")
    print("=" * 60)
    print("        SPEECH EMOTION RECOGNITION SYSTEM")
    print("=" * 60)

    print(f"\nAudio File:")
    print(file_path)

    print("\nPredicted Emotion:")
    print(f"  {predicted_emotion.upper()}")

    print("\nConfidence:")
    print(f"  {confidence * 100:.2f}%")

    print("\nEmotion Probabilities:")
    print("-" * 40)

    # Sort probabilities from highest to lowest
    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for emotion, probability in sorted_probabilities:

        print(
            f"{emotion:<12} "
            f"{probability * 100:>7.2f}%"
        )

    print("=" * 60)
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Check command-line argument
    if len(sys.argv) < 2:

        print("\nUsage:")
        print("python predict_audio.py <audio_file.wav>")

        print("\nExample:")
        print(
            "python predict_audio.py "
            "dataset/RAVDESS/audio_speech_actors_01-24/"
            "Actor_01/03-01-05-01-02-01-12.wav"
        )

        sys.exit(1)

    audio_file = sys.argv[1]

    try:

        print("\nLoading trained model...")
        print("Extracting audio features...")
        print("Running emotion prediction...")

        (
            predicted_emotion,
            confidence,
            probabilities
        ) = predict_emotion(audio_file)

        display_prediction(
            audio_file,
            predicted_emotion,
            confidence,
            probabilities
        )

    except Exception as e:

        print("\nERROR:")
        print(str(e))
        sys.exit(1)