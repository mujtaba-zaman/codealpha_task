import os
import numpy as np
from dataset_loader import load_dataset
from feature_extraction import extract_mfcc

OUTPUT_DIR = "dataset/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = load_dataset()

X = []
y = []

emotion_labels = sorted(df["emotion"].unique())
label_map = {emotion: i for i, emotion in enumerate(emotion_labels)}

print("Processing audio files...")

for index, row in df.iterrows():

    try:
        mfcc = extract_mfcc(row["file_path"])

        X.append(mfcc)
        y.append(label_map[row["emotion"]])

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(df)}")

    except Exception as e:
        print(f"Error: {row['file_path']}")
        print(e)

X = np.array(X)
y = np.array(y)

# CNN input: samples, height, width, channel
X = X[..., np.newaxis]

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), np.array(emotion_labels))

print("\nData preparation completed!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", emotion_labels)