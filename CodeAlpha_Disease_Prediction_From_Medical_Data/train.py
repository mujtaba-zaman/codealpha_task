import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import joblib
import os


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("DISEASE PREDICTION FROM MEDICAL DATA")
print("=" * 60)

dataset_path = os.path.join("dataset", "heart_disease.csv")

df = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(df["target_binary"].value_counts())

print("\nTarget meaning:")
print("0 = No Heart Disease")
print("1 = Heart Disease")


# ============================================================
# 4. REMOVE UNNECESSARY / TARGET-RELATED COLUMN
# ============================================================

# 'num' is the original disease severity/target information.
# We do NOT use it as an input feature because target_binary
# was derived from this information.

X = df.drop(columns=["target_binary", "num"])
y = df["target_binary"]

print("\n" + "=" * 60)
print("FEATURES AND TARGET")
print("=" * 60)

print("\nFeatures:")
print(X.columns.tolist())

print(f"\nNumber of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 6. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            probability=True,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# ============================================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================================

results = {}

print("\n" + "=" * 60)
print("MODEL TRAINING AND EVALUATION")
print("=" * 60)

for model_name, model in models.items():

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    # Train
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Probability
    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)

    # Store results
    results[model_name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Heart Disease", "Heart Disease"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison_data = []

for model_name, result in results.items():

    comparison_data.append({
        "Model": model_name,
        "Accuracy": result["accuracy"],
        "Precision": result["precision"],
        "Recall": result["recall"],
        "F1-Score": result["f1"],
        "ROC-AUC": result["roc_auc"]
    })

comparison_df = pd.DataFrame(comparison_data)

print(
    comparison_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC"
        ]
    ].to_string(index=False)
)


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_model_name = max(
    results,
    key=lambda model_name: results[model_name]["roc_auc"]
)

best_model = results[best_model_name]["model"]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"\nBest Model: {best_model_name}")
print(f"Best ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")


# ============================================================
# 10. SAVE BEST MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model_path = os.path.join(
    "models",
    "best_disease_prediction_model.pkl"
)

joblib.dump(best_model, model_path)

print("\nModel saved successfully!")
print(f"Location: {model_path}")


# ============================================================
# 11. SAVE FEATURE NAMES
# ============================================================

feature_path = os.path.join(
    "models",
    "feature_names.pkl"
)

joblib.dump(X.columns.tolist(), feature_path)

print(f"Feature names saved: {feature_path}")


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nNext step:")
print("We will create the Streamlit interface using app.py.")