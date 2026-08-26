import os
import sys
import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/emotion_cnn.keras"
LABELS_PATH = "dataset/processed/labels.npy"
CONFUSION_MATRIX_PATH = "models/confusion_matrix.png"

SAMPLE_RATE = 22050
N_MFCC = 40
MFCC_WIDTH = 174


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}

/* Main title */

.hero-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.hero-subtitle {
    text-align: center;
    font-size: 19px;
    opacity: 0.75;
    margin-bottom: 35px;
}

/* Cards */

.info-card {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 20px;
}

.result-card {
    padding: 30px;
    border-radius: 18px;
    border: 2px solid rgba(128,128,128,0.3);
    text-align: center;
    margin-top: 20px;
}

.result-emotion {
    font-size: 42px;
    font-weight: 800;
    margin-top: 10px;
}

.confidence {
    font-size: 25px;
    font-weight: 600;
}

/* Section heading */

.section-title {
    font-size: 30px;
    font-weight: 750;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Feature cards */

.feature-card {
    padding: 22px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    min-height: 150px;
}

/* Footer */

.footer {
    text-align: center;
    opacity: 0.65;
    margin-top: 50px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_emotion_model():

    model = load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH)

    return model, labels


# ============================================================
# EXTRACT MFCC
# ============================================================

def extract_mfcc(audio_file):

    audio, sample_rate = librosa.load(
        audio_file,
        sr=SAMPLE_RATE
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC
    )

    mfcc = librosa.util.fix_length(
        mfcc,
        size=MFCC_WIDTH,
        axis=1
    )

    mfcc = mfcc.astype("float32")

    mfcc = (
        mfcc - mfcc.mean()
    ) / (
        mfcc.std() + 1e-8
    )

    mfcc = mfcc[..., np.newaxis]

    mfcc = np.expand_dims(
        mfcc,
        axis=0
    )

    return mfcc


# ============================================================
# PREDICTION
# ============================================================

def predict_emotion(audio_file):

    model, labels = load_emotion_model()

    features = extract_mfcc(audio_file)

    predictions = model.predict(
        features,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = np.argmax(
        probabilities
    )

    emotion = str(
        labels[predicted_index]
    )

    confidence = float(
        probabilities[predicted_index]
    )

    return emotion, confidence, probabilities, labels


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎙️ SER System")

    st.markdown("---")

    st.markdown("### Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Home",
            "🎙️ Emotion Prediction",
            "📊 Model Performance",
            "📚 About Project"
        ]
    )

    st.markdown("---")

    st.markdown("### Model")

    st.write("**Architecture:** CNN")
    st.write("**Features:** MFCC")
    st.write("**Classes:** 8")
    st.write("**Test Accuracy:** 55.56%")

    st.markdown("---")

    st.caption(
        "Speech Emotion Recognition System"
    )


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="hero-title">'
        '🎙️ Speech Emotion Recognition'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'CNN-Based Speech Emotion Detection System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

        ### Welcome 👋

        This project is a **Speech Emotion Recognition (SER)**
        system that uses **Machine Learning and Deep Learning**
        techniques to identify emotions from human speech.

        The system extracts **MFCC audio features** from speech
        and uses a **Convolutional Neural Network (CNN)** to
        classify the speech into one of eight emotions.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">✨ Key Features</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="feature-card">

            ### 🎧 Audio Input

            Upload a WAV speech file
            for emotion analysis.

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">

            ### 🧠 CNN Model

            Deep learning model trained
            for speech emotion classification.

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">

            ### 🎵 MFCC Features

            Audio is converted into
            meaningful MFCC features.

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="feature-card">

            ### 📈 Prediction

            Shows predicted emotion
            and confidence score.

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🎭 Supported Emotions</div>',
        unsafe_allow_html=True
    )

    emotions = [
        "Angry",
        "Calm",
        "Disgust",
        "Fearful",
        "Happy",
        "Neutral",
        "Sad",
        "Surprised"
    ]

    cols = st.columns(8)

    for col, emotion in zip(cols, emotions):

        with col:
            st.metric(
                label="Emotion",
                value=emotion
            )

    st.markdown("---")

    st.info(
        "Go to **Emotion Prediction** from the sidebar "
        "to upload an audio file and test the system."
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🎙️ Emotion Prediction":

    st.markdown(
        '<div class="hero-title">'
        '🎙️ Emotion Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Upload a speech recording and let the CNN model '
        'predict the emotional state.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload Speech Audio",
        type=["wav"],
        help="Currently supported format: WAV"
    )

    if uploaded_file is not None:

        st.success(
            f"Audio uploaded successfully: "
            f"{uploaded_file.name}"
        )

        st.markdown("### 🎧 Audio Preview")

        st.audio(
            uploaded_file,
            format="audio/wav"
        )

        st.markdown("---")

        if st.button(
            "🔍 Analyze Emotion",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing speech and predicting emotion..."
            ):

                try:

                    (
                        emotion,
                        confidence,
                        probabilities,
                        labels
                    ) = predict_emotion(
                        uploaded_file
                    )

                    st.markdown(
                        f"""
                        <div class="result-card">

                        <div>Predicted Emotion</div>

                        <div class="result-emotion">
                        {emotion.upper()}
                        </div>

                        <div class="confidence">
                        Confidence: {confidence * 100:.2f}%
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("---")

                    st.markdown(
                        '<div class="section-title">'
                        '📊 Emotion Probabilities'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    probability_data = {
                        str(label): float(prob)
                        for label, prob
                        in zip(labels, probabilities)
                    }

                    sorted_data = dict(
                        sorted(
                            probability_data.items(),
                            key=lambda item: item[1],
                            reverse=True
                        )
                    )

                    for label, probability in sorted_data.items():

                        st.write(
                            f"**{label.capitalize()}**"
                        )

                        st.progress(
                            float(probability)
                        )

                        st.caption(
                            f"{probability * 100:.2f}%"
                        )

                except Exception as e:

                    st.error(
                        f"Prediction failed: {str(e)}"
                    )

    else:

        st.info(
            "👆 Upload a WAV audio file above "
            "to start emotion prediction."
        )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="hero-title">'
        '📊 Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Evaluation results of the trained CNN model'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Test Accuracy",
            "55.56%"
        )

    with col2:
        st.metric(
            "Test Loss",
            "1.2421"
        )

    with col3:
        st.metric(
            "Emotion Classes",
            "8"
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🎯 Classification Performance'
        '</div>',
        unsafe_allow_html=True
    )

    performance_data = {
        "Emotion": [
            "Angry",
            "Calm",
            "Disgust",
            "Fearful",
            "Happy",
            "Neutral",
            "Sad",
            "Surprised"
        ],

        "Precision": [
            0.5517,
            0.6667,
            0.6875,
            0.5833,
            0.3529,
            0.0000,
            0.3200,
            0.8095
        ],

        "Recall": [
            0.8421,
            0.4211,
            0.5789,
            0.7000,
            0.3158,
            0.0000,
            0.4211,
            0.8947
        ],

        "F1 Score": [
            0.6667,
            0.5161,
            0.6286,
            0.6364,
            0.3333,
            0.0000,
            0.3636,
            0.8500
        ]
    }

    st.dataframe(
        performance_data,
        use_container_width=True
    )

    st.markdown("---")

    if os.path.exists(
        CONFUSION_MATRIX_PATH
    ):

        st.markdown(
            '<div class="section-title">'
            '🔲 Confusion Matrix'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            CONFUSION_MATRIX_PATH,
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix image was not found."
        )


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif page == "📚 About Project":

    st.markdown(
        '<div class="hero-title">'
        '📚 About the Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Project information, technologies and methodology'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

        ## 🎯 Project Objective

        The main objective of this project is to develop a
        deep learning based system capable of recognizing
        human emotions from speech audio.

        The system processes speech recordings, extracts
        MFCC features and uses a CNN classifier to predict
        the emotional category.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "## 🗂️ Dataset"
    )

    st.write(
        """
        **Dataset:** RAVDESS

        **Total unique speech files used:** 1,440

        **Actors:** 24

        **Emotion classes:** 8

        **Audio format:** WAV

        **Sampling rate:** 48 kHz original audio

        The eight emotion categories are:

        - Angry
        - Calm
        - Disgust
        - Fearful
        - Happy
        - Neutral
        - Sad
        - Surprised
        """
    )

    st.markdown("---")

    st.markdown(
        "## 🛠️ Libraries & Technologies"
    )

    libraries = {
        "Technology": [
            "Python",
            "TensorFlow / Keras",
            "Librosa",
            "NumPy",
            "Pandas",
            "Scikit-learn",
            "Matplotlib",
            "Seaborn",
            "Streamlit"
        ],

        "Purpose": [
            "Main programming language",
            "CNN model development and training",
            "Audio processing and MFCC extraction",
            "Numerical data processing",
            "Dataset management",
            "Data splitting and model evaluation",
            "Visualization",
            "Confusion matrix visualization",
            "Web application and UI"
        ]
    }

    st.table(
        libraries
    )

    st.markdown("---")

    st.markdown(
        "## 🔄 Project Workflow"
    )

    workflow = [
        "1. RAVDESS Dataset",
        "2. Audio Loading",
        "3. MFCC Feature Extraction",
        "4. Data Preparation",
        "5. CNN Model Training",
        "6. Model Evaluation",
        "7. Saved Trained Model",
        "8. Audio Upload",
        "9. Emotion Prediction"
    ]

    for step in workflow:
        st.write(step)

    st.markdown("---")

    st.markdown(
        "## 🧠 Model Architecture"
    )

    st.write(
        """
        The project uses a Convolutional Neural Network (CNN)
        for classification.

        **Input:** 40 × 174 × 1 MFCC representation

        **Convolutional Layers:** 32, 64 and 128 filters

        **Pooling:** Max Pooling

        **Regularization:** Dropout

        **Normalization:** Batch Normalization

        **Dense Layer:** 128 neurons

        **Output:** 8-class Softmax classification
        """
    )

    st.markdown("---")

    st.markdown(
        "## 📌 Current Result"
    )

    st.success(
        "Final Test Accuracy: 55.56%"
    )

    st.markdown(
        """
        The current trained CNN provides a baseline
        speech emotion recognition system. The Streamlit
        interface allows the trained model to be tested
        using individual WAV speech recordings.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🎙️ **Speech Emotion Recognition System**  
    CNN-Based Audio Emotion Classification

    Developed using Python, TensorFlow, Librosa and Streamlit

    </div>
    """,
    unsafe_allow_html=True
)