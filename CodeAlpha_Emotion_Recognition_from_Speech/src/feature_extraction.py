import librosa
import numpy as np
from dataset_loader import load_dataset


def extract_mfcc(file_path, n_mfcc=40):
    audio, sample_rate = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc
    )

    # Fixed size for CNN
    mfcc = librosa.util.fix_length(mfcc, size=174, axis=1)

    return mfcc


if __name__ == "__main__":

    df = load_dataset()

    sample_file = df.iloc[0]["file_path"]
    sample_emotion = df.iloc[0]["emotion"]

    mfcc = extract_mfcc(sample_file)

    print("Feature extraction successful!")
    print("Audio:", sample_file)
    print("Emotion:", sample_emotion)
    print("MFCC Shape:", mfcc.shape)