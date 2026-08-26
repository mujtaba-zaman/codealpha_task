import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Heart Disease AI Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #f7f7f5;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #151515;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f2c94c !important;
    }

    section[data-testid="stSidebar"] p {
        color: #dddddd !important;
    }

    .sidebar-label {
        color: #f2c94c;
        font-size: 13px;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 2px;
    }

    .sidebar-value {
        color: #eeeeee;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .sidebar-description {
        color: #bdbdbd;
        font-size: 13px;
        line-height: 1.5;
    }

    .sidebar-line {
        border-bottom: 1px solid #333333;
        margin: 20px 0;
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    h1, h2, h3 {
        color: #111111 !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        background-color: #111111;
        color: white;
        border: 2px solid #f2c94c;
        border-radius: 10px;
        height: 52px;
        font-size: 17px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #f2c94c;
        color: #111111;
        border-color: #111111;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #666666 !important;
    }

    [data-testid="stMetricValue"] {
        color: #111111 !important;
    }


    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    .stProgress > div > div > div > div {
        background-color: #f2c94c;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "best_disease_prediction_model.pkl"
)

FEATURE_PATH = os.path.join(
    "models",
    "feature_names.pkl"
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "Model file not found: "
        "models/best_disease_prediction_model.pkl"
    )

    st.stop()


if not os.path.exists(FEATURE_PATH):

    st.error(
        "Feature file not found: "
        "models/feature_names.pkl"
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

    feature_names = joblib.load(FEATURE_PATH)

except Exception as e:

    st.error(
        f"Error loading model: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🩺 System Information"
    )

    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-label">Application</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-value">'
        'Heart Disease Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-label">Task</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-value">'
        'Binary Classification'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-label">Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-value">'
        'Heart Disease / No Heart Disease'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-label">Probability</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-value">'
        'Model Prediction Probability'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-label">Features</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-value">'
        '13 Medical Features'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 🤖 Saved Model"
    )

    st.markdown(
        '<div class="sidebar-description">'
        'Best performing model selected using ROC-AUC.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### ⚠️ Disclaimer"
    )

    st.markdown(
        '<div class="sidebar-description">'
        'This application is developed for educational and '
        'machine learning purposes. The prediction should not '
        'be considered a medical diagnosis.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "❤️ Heart Disease AI Prediction"
)

st.caption(
    "Machine learning based cardiovascular risk prediction "
    "using patient medical information."
)

st.divider()


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header(
    "👤 Patient Information"
)

st.write(
    "Enter the patient's medical information below "
    "to generate a prediction."
)


# ============================================================
# ROW 1
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=45,
        step=1
    )

with col2:

    sex = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x:
        "Female" if x == 0 else "Male"
    )

with col3:

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3]
    )


# ============================================================
# ROW 2
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120,
        step=1
    )

with col2:

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=200,
        step=1
    )

with col3:

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


# ============================================================
# ROW 3
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    restecg = st.selectbox(
        "Resting ECG",
        [0, 1, 2]
    )

with col2:

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )

with col3:

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


# ============================================================
# ROW 4
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

with col2:

    slope = st.selectbox(
        "ST Segment Slope",
        [0, 1, 2]
    )

with col3:

    ca = st.selectbox(
        "Major Vessels (CA)",
        [0, 1, 2, 3, 4]
    )


# ============================================================
# ROW 5
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    thal = st.selectbox(
        "Thalassemia",
        [0, 1, 2, 3]
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")

left, center, right = st.columns(
    [1, 2, 1]
)

with center:

    predict_button = st.button(
        "🔍 Predict Disease",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]],
            columns=feature_names
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            input_data
        )[0]

        no_disease_probability = (
            probabilities[0] * 100
        )

        disease_probability = (
            probabilities[1] * 100
        )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.divider()

        st.header(
            "📊 Prediction Result"
        )


        if prediction == 1:

            st.error(
                "⚠️ Heart Disease Detected"
            )

            st.write(
                "The trained machine learning model predicts "
                "a positive heart disease result."
            )

        else:

            st.success(
                "✅ No Heart Disease Detected"
            )

            st.write(
                "The trained machine learning model predicts "
                "a negative heart disease result."
            )


        # ====================================================
        # PROBABILITY
        # ====================================================

        st.header(
            "🎯 Prediction Probability"
        )


        probability_col1, probability_col2 = st.columns(2)


        with probability_col1:

            st.metric(
                "No Heart Disease",
                f"{no_disease_probability:.2f}%"
            )

            st.progress(
                min(
                    int(no_disease_probability),
                    100
                )
            )


        with probability_col2:

            st.metric(
                "Heart Disease",
                f"{disease_probability:.2f}%"
            )

            st.progress(
                min(
                    int(disease_probability),
                    100
                )
            )


        # ====================================================
        # PATIENT SUMMARY
        # ====================================================

        st.header(
            "📋 Patient Summary"
        )


        summary1, summary2, summary3, summary4 = st.columns(4)


        with summary1:

            st.metric(
                "Age",
                f"{age} years"
            )


        with summary2:

            st.metric(
                "Blood Pressure",
                f"{trestbps} mmHg"
            )


        with summary3:

            st.metric(
                "Cholesterol",
                f"{chol} mg/dl"
            )


        with summary4:

            st.metric(
                "Maximum Heart Rate",
                str(thalach)
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.header(
            "🤖 Model Information"
        )


        model1, model2, model3 = st.columns(3)


        with model1:

            st.metric(
                "Input Features",
                len(feature_names)
            )


        with model2:

            st.metric(
                "Classes",
                "2"
            )


        with model3:

            st.metric(
                "Probability",
                "Available"
            )


        # ====================================================
        # MEDICAL DISCLAIMER
        # ====================================================

        st.warning(
            "Medical Disclaimer: This prediction is generated "
            "by a machine learning model for educational and "
            "demonstration purposes. It should not be used as "
            "a substitute for professional medical examination, "
            "diagnosis, or treatment."
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )