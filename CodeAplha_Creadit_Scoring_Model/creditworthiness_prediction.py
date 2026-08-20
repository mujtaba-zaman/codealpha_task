import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

train_path = os.path.join(
    DATA_DIR,
    "train-FIN_ANA_DATA.xlsx"
)

test_path = os.path.join(
    DATA_DIR,
    "test-FIN_ANA_DATA.xlsx"
)

print("=" * 60)
print("CHECKING FILE PATHS")
print("=" * 60)

print("Train file:", train_path)
print("Train file exists:", os.path.exists(train_path))

print("Test file:", test_path)
print("Test file exists:", os.path.exists(test_path))


# Stop program if files are not found

if not os.path.exists(train_path):
    raise FileNotFoundError(
        f"Training file not found:\n{train_path}"
    )

if not os.path.exists(test_path):
    raise FileNotFoundError(
        f"Test file not found:\n{test_path}"
    )


# ============================================================
# 2. LOAD DATA
# ============================================================

train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Training Shape:", train_df.shape)
print("Testing Shape :", test_df.shape)

print("\nTraining Columns:")
print(train_df.columns.tolist())


# ============================================================
# 3. BASIC DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 TRAINING RECORDS")
print("=" * 60)

print(train_df.head())


print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(train_df.dtypes)


print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(train_df.isnull().sum())


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(train_df["QUALITY_OF_LOAN"].value_counts())

print("\nPercentage:")
print(
    train_df["QUALITY_OF_LOAN"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 5. TARGET VARIABLE
# ============================================================

# QUALITY_OF_LOAN:
#
# G = Good
# B = Bad
#
# Machine Learning target:
#
# Good = 0
# Bad  = 1

train_df["TARGET"] = train_df["QUALITY_OF_LOAN"].map({
    "G": 0,
    "B": 1
})

print("\n" + "=" * 60)
print("TARGET MAPPING")
print("=" * 60)

print("G = 0 (Good Loan Quality)")
print("B = 1 (Bad Loan Quality)")


# Check for unexpected target values

if train_df["TARGET"].isnull().any():
    print("\nWARNING: Unexpected values found in QUALITY_OF_LOAN")


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    # Avoid division by zero

    investment = df["INVESTMENT_TOTAL"].replace(
        0,
        np.nan
    )

    balance = df["ACCCURRENTBALANCE"].replace(
        0,
        np.nan
    )

    # --------------------------------------------------------
    # Balance to Investment Ratio
    # --------------------------------------------------------

    df["BALANCE_TO_INVESTMENT"] = (
        df["ACCCURRENTBALANCE"] / investment
    )

    # --------------------------------------------------------
    # Due Payment to Investment Ratio
    # --------------------------------------------------------

    df["DUE_TO_INVESTMENT"] = (
        df["DUE_PAYMENT"] / investment
    )

    # --------------------------------------------------------
    # Due Payment to Current Balance Ratio
    # --------------------------------------------------------

    df["DUE_TO_BALANCE"] = (
        df["DUE_PAYMENT"] / balance
    )

    return df


train_df = create_features(train_df)

test_df = create_features(test_df)


# ============================================================
# 7. PREPARE FEATURES AND TARGET
# ============================================================

# ACC_NO is an account identifier.
# It should NOT be used as a predictive feature.

X = train_df.drop(
    columns=[
        "QUALITY_OF_LOAN",
        "TARGET",
        "ACC_NO"
    ]
)

y = train_df["TARGET"]


# Save test account numbers

test_ids = test_df["ACC_NO"]


# Remove account number from test features

X_test_final = test_df.drop(
    columns=["ACC_NO"]
)


# ============================================================
# 8. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\n" + "=" * 60)
print("FEATURE INFORMATION")
print("=" * 60)

print("\nNumerical Features:")
for feature in numeric_features:
    print("-", feature)

print("\nCategorical Features:")
for feature in categorical_features:
    print("-", feature)


# ============================================================
# 8.1 FIX MIXED DATA TYPES
# ============================================================

# Some categorical columns contain a mixture of
# numbers and strings. Convert all categorical values
# to strings so OneHotEncoder can process them safely.

for column in categorical_features:

    X[column] = X[column].astype(str)
    X_test_final[column] = X_test_final[column].astype(str)


# ============================================================
# 9. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / VALIDATION SPLIT")
print("=" * 60)

print("Training records  :", len(X_train))
print("Validation records:", len(X_valid))


# ============================================================
# 10. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 11. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=20,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 12. TRAIN AND EVALUATE MODELS
# ============================================================

results = {}

trained_models = {}


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # Create pipeline

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_valid
    )

    y_probability = pipeline.predict_proba(
        X_valid
    )[:, 1]


    # --------------------------------------------------------
    # Evaluation Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_valid,
        y_pred
    )

    precision = precision_score(
        y_valid,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_valid,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_valid,
        y_probability
    )


    # Save results

    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }

    trained_models[model_name] = pipeline


    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_valid,
            y_pred,
            target_names=[
                "Good",
                "Bad"
            ],
            zero_division=0
        )
    )


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
).T


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.round(4)
)


# Save model comparison

comparison_path = os.path.join(
    BASE_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_path
)

print(
    "\nModel comparison saved to:",
    comparison_path
)


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

# ROC-AUC is used for selecting the best model.

best_model_name = results_df[
    "ROC-AUC"
].idxmax()

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Best Model:",
    best_model_name
)

print(
    "Best ROC-AUC:",
    round(
        results_df.loc[
            best_model_name,
            "ROC-AUC"
        ],
        4
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

best_predictions = best_model.predict(
    X_valid
)


cm = confusion_matrix(
    y_valid,
    best_predictions
)


plt.figure(
    figsize=(6, 5)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Good",
        "Bad"
    ],
    yticklabels=[
        "Good",
        "Bad"
    ]
)


plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()


confusion_path = os.path.join(
    BASE_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300
)

plt.show()


print(
    "\nConfusion matrix saved to:",
    confusion_path
)


# ============================================================
# 16. ROC CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)


for model_name, pipeline in trained_models.items():

    probabilities = pipeline.predict_proba(
        X_valid
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_valid,
        probabilities
    )

    auc_score = roc_auc_score(
        y_valid,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=(
            f"{model_name} "
            f"(AUC = {auc_score:.3f})"
        )
    )


# Random classifier line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Model Comparison"
)

plt.legend()

plt.tight_layout()


roc_path = os.path.join(
    BASE_DIR,
    "roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300
)

plt.show()


print(
    "ROC curve saved to:",
    roc_path
)


# ============================================================
# 17. TRAIN BEST MODEL ON COMPLETE DATASET
# ============================================================

print("\n" + "=" * 60)
print("TRAINING BEST MODEL ON FULL DATA")
print("=" * 60)

best_model.fit(
    X,
    y
)

print(
    "Best model trained successfully."
)


# ============================================================
# 18. PREDICT TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("PREDICTING TEST DATA")
print("=" * 60)


test_predictions = best_model.predict(
    X_test_final
)


test_probabilities = best_model.predict_proba(
    X_test_final
)[:, 1]


# Convert numerical predictions back to G/B

predicted_quality = np.where(
    test_predictions == 0,
    "G",
    "B"
)


# ============================================================
# 19. CREATE FINAL PREDICTION DATAFRAME
# ============================================================

prediction_df = pd.DataFrame({

    "ACC_NO": test_ids,

    "PREDICTED_QUALITY_OF_LOAN":
        predicted_quality,

    "BAD_LOAN_PROBABILITY":
        test_probabilities

})


# ============================================================
# 20. SAVE TEST PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    BASE_DIR,
    "test_predictions.csv"
)


prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)

print(
    "\nFirst 10 predictions:"
)

print(
    prediction_df.head(10)
)


print(
    "\nTotal predictions:",
    len(prediction_df)
)


print(
    "\nPrediction file saved to:"
)

print(
    prediction_path
)


print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)