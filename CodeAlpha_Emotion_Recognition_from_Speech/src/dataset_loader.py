import os
import pandas as pd

# Path to the actual RAVDESS speech dataset
DATASET_PATH = "dataset/RAVDESS/audio_speech_actors_01-24"

# RAVDESS emotion codes
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def load_dataset():
    data = []

    for actor_folder in sorted(os.listdir(DATASET_PATH)):

        actor_path = os.path.join(DATASET_PATH, actor_folder)

        if not os.path.isdir(actor_path):
            continue

        for filename in os.listdir(actor_path):

            if not filename.lower().endswith(".wav"):
                continue

            # Example filename:
            # 03-01-05-01-02-01-12.wav

            parts = filename.split("-")

            emotion_code = parts[2]

            if emotion_code in EMOTIONS:

                emotion = EMOTIONS[emotion_code]

                file_path = os.path.join(actor_path, filename)

                data.append({
                    "file_path": file_path,
                    "emotion": emotion,
                    "actor": actor_folder
                })

    return pd.DataFrame(data)


if __name__ == "__main__":

    df = load_dataset()

    print("\nDataset loaded successfully!")
    print("--------------------------------")
    print("Total audio files:", len(df))

    print("\nEmotion distribution:")
    print(df["emotion"].value_counts().sort_index())

    print("\nFirst 10 records:")
    print(df.head(10))