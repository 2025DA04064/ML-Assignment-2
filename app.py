"""
Streamlit app for the ML Assignment 2 submission 
Student ID - 2025DA04064
Student Name - Niraj Sonawane

Features:
  a. CSV upload of test data
  b. Model selection dropdown (5 classifiers)
  c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix + classification report

Run locally:   streamlit run app.py
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classifier Comparison",
    page_icon="🩺",
    layout="wide",
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model", "saved_models")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}


@st.cache_resource
def load_artifacts():
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in MODEL_FILES.items()
    }
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)
    return models, scaler, meta


@st.cache_data
def load_default_test_data():
    return pd.read_csv(os.path.join(HERE, "test_data.csv"))


models, scaler, meta = load_artifacts()
feature_names = meta["feature_names"]
target_names = meta["target_names"]  # ['malignant', 'benign']
scaled_models = set(meta["scaled_models"])

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

st.sidebar.subheader("1. Test data")
uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV). Must contain the 30 numeric features "
    "and a 'target' column (0 = malignant, 1 = benign).",
    type=["csv"],
)
use_default = st.sidebar.checkbox("Use sample test_data.csv", value=uploaded_file is None)

st.sidebar.subheader("2. Model")
selected_model_name = st.sidebar.selectbox("Choose a classification model", list(models.keys()))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🩺 Breast Cancer Classification — Model Comparison App")
st.caption(
    "Dataset: Breast Cancer Wisconsin (Diagnostic), 569 instances, 30 features. "
    "5 classifiers trained: Logistic Regression, Decision Tree, kNN, Naive Bayes, "
    "Random Forest (Ensemble)."
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if uploaded_file is not None and not use_default:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded uploaded file: {uploaded_file.name}")
elif uploaded_file is not None and use_default:
    df = pd.read_csv(uploaded_file)
    st.sidebar.info("Using uploaded file (checkbox override).")
else:
    df = load_default_test_data()
    st.sidebar.info("Using sample test_data.csv")

missing_cols = [c for c in feature_names if c not in df.columns]
if missing_cols:
    st.error(
        f"Uploaded file is missing {len(missing_cols)} required feature column(s), "
        f"e.g. {missing_cols[:5]}. Please upload a CSV matching the expected schema."
    )
    st.stop()

has_target = "target" in df.columns

with st.expander("🔍 Preview uploaded / sample data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

X = df[feature_names]
y_true = df["target"] if has_target else None

# ---------------------------------------------------------------------------
# Predict with selected model
# ---------------------------------------------------------------------------
model = models[selected_model_name]
X_input = scaler.transform(X) if selected_model_name in scaled_models else X.values

y_pred = model.predict(X_input)
y_proba = model.predict_proba(X_input)[:, 1]

st.subheader(f"Results — {selected_model_name}")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Predictions (first 15 rows)**")
    pred_df = pd.DataFrame(
        {
            "Predicted class": [target_names[p] for p in y_pred],
            "P(benign)": np.round(y_proba, 4),
        }
    )
    if has_target:
        pred_df.insert(0, "True class", [target_names[t] for t in y_true])
    st.dataframe(pred_df.head(15), use_container_width=True)

with col2:
    if has_target:
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        st.markdown("**Evaluation metrics on this test data**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("AUC", f"{auc:.4f}")
        m3.metric("Precision", f"{prec:.4f}")
        m4, m5, m6 = st.columns(3)
        m4.metric("Recall", f"{rec:.4f}")
        m5.metric("F1 Score", f"{f1:.4f}")
        m6.metric("MCC", f"{mcc:.4f}")
    else:
        st.info(
            "Uploaded CSV has no 'target' column, so evaluation metrics and the "
            "confusion matrix can't be computed — predictions only."
        )

# ---------------------------------------------------------------------------
# Confusion matrix + classification report
# ---------------------------------------------------------------------------
if has_target:
    st.subheader("Confusion Matrix & Classification Report")
    c1, c2 = st.columns([1, 1.2])

    with c1:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {selected_model_name}")
        st.pyplot(fig)

    with c2:
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        )
        report_df = pd.DataFrame(report).transpose().round(4)
        st.markdown("**Classification Report**")
        st.dataframe(report_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Model comparison table (across all 5 models on the same data)
# ---------------------------------------------------------------------------
if has_target:
    st.subheader("📊 Compare All 5 Models on This Test Data")
    rows = []
    for name, mdl in models.items():
        Xi = scaler.transform(X) if name in scaled_models else X.values
        yp = mdl.predict(Xi)
        ypr = mdl.predict_proba(Xi)[:, 1]
        rows.append(
            {
                "ML Model Name": name,
                "Accuracy": accuracy_score(y_true, yp),
                "AUC": roc_auc_score(y_true, ypr),
                "Precision": precision_score(y_true, yp),
                "Recall": recall_score(y_true, yp),
                "F1": f1_score(y_true, yp),
                "MCC": matthews_corrcoef(y_true, yp),
            }
        )
    comp_df = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    st.dataframe(comp_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

st.markdown("---")
st.caption(
    "Built for ML Assignment 2 by Niraj Sonawane (2025DA04064) — Streamlit app deployed on Streamlit Community Cloud."
)
