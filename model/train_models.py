"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates them, saves the fitted models + scaler to disk, and
writes out:
  - test_data.csv        (held-out test split, used by the Streamlit app)
  - model/metrics.csv     (comparison table of evaluation metrics)

Dataset: Breast Cancer Wisconsin (Diagnostic) Data Set
Source:  scikit-learn built-in loader (originally UCI ML Repository)
         https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
Shape:   569 instances, 30 numeric features, binary target
         (0 = malignant, 1 = benign)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
feature_names = list(X.columns)
target_names = list(data.target_names)  # ['malignant', 'benign']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 2. Scale features (helps Logistic Regression / kNN especially)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Save the raw test split as test_data.csv for the Streamlit app
#    (features + true label, so the app can score predictions)
# ---------------------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

# ---------------------------------------------------------------------------
# 4. Define the 5 required models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
}

# Models that need scaled input for best behaviour
SCALED_MODELS = {"Logistic Regression", "kNN"}

results = []
os.makedirs(os.path.join(HERE, "saved_models"), exist_ok=True)

for name, model in models.items():
    if name in SCALED_MODELS:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results.append(metrics)

    # Save each fitted model
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, os.path.join(HERE, "saved_models", f"{safe_name}.joblib"))
    print(f"Trained {name}: acc={metrics['Accuracy']:.4f} auc={metrics['AUC']:.4f}")

# Save the scaler too (needed at inference time for LR / kNN)
joblib.dump(scaler, os.path.join(HERE, "saved_models", "scaler.joblib"))

# Save feature/target metadata so app.py doesn't need to touch sklearn.datasets
meta = {"feature_names": feature_names, "target_names": target_names,
        "scaled_models": list(SCALED_MODELS)}
with open(os.path.join(HERE, "saved_models", "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

# ---------------------------------------------------------------------------
# 5. Save comparison table
# ---------------------------------------------------------------------------
metrics_df = pd.DataFrame(results)
metrics_df.to_csv(os.path.join(HERE, "metrics.csv"), index=False)
print("\nComparison table:\n", metrics_df.round(4).to_string(index=False))
print("\nSaved models to model/saved_models/, metrics to model/metrics.csv,"
      " and test split to test_data.csv")
