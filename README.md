# Breast Cancer Classification — ML Model Comparison & Streamlit App

## a. Problem Statement

Breast cancer diagnosis relies on interpreting features extracted from digitized
images of a fine needle aspirate (FNA) of a breast mass. The goal of this project
is to build and compare multiple classification models that predict whether a
breast mass is **malignant** or **benign** from a set of numeric diagnostic
features, and to expose the trained models through an interactive Streamlit web
application so predictions and evaluation metrics can be explored on demand.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** Loaded via `sklearn.datasets.load_breast_cancer()`, which packages
  the classic UCI ML Repository dataset
  (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).
- **Instances:** 569 (meets the ≥500 instance requirement)
- **Features:** 30 numeric features (meets the ≥12 feature requirement) —
  mean, standard-error, and "worst" values of 10 real-valued measurements
  computed from each cell nucleus (radius, texture, perimeter, area,
  smoothness, compactness, concavity, concave points, symmetry, fractal
  dimension).
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant / 357 benign
- **Split used:** 75% train / 25% test, stratified on the target
  (`random_state=42`)
- `test_data.csv` in this repo is the held-out **test split** (features +
  true `target` column) used both for offline evaluation and as the default
  sample data inside the Streamlit app.

## c. GitHub Repository Link

> **TODO:** Replace with your actual GitHub repo URL after you push this project,
> e.g. `https://github.com/<your-username>/<repo-name>`

## d. Models Used

All 5 models below were trained on the **same** dataset and train/test split.
Logistic Regression and kNN were trained on `StandardScaler`-scaled features
(distance/coefficient-based models); Decision Tree, Naive Bayes, and Random
Forest were trained on the raw features (tree/probability-based models don't
need scaling).

### Comparison Table

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.9860   | 0.9977 | 0.9889    | 0.9889 | 0.9889 | 0.9700 |
| Decision Tree               | 0.9231   | 0.9234 | 0.9540    | 0.9222 | 0.9379 | 0.8378 |
| kNN                         | 0.9790   | 0.9845 | 0.9677    | 1.0000 | 0.9836 | 0.9555 |
| Naive Bayes                 | 0.9371   | 0.9893 | 0.9263    | 0.9778 | 0.9514 | 0.8650 |
| Random Forest (Ensemble)    | 0.9580   | 0.9949 | 0.9565    | 0.9778 | 0.9670 | 0.9098 |

*(Regenerate this table any time by re-running `python model/train_models.py`,
which writes fresh numbers to `model/metrics.csv`.)*

### Observations

| ML Model Name             | Observation about model performance |
|-----------------------------|--------------------------------------|
| Logistic Regression        | Best all-round performer here — the classes are close to linearly separable in this feature space, so a linear decision boundary on scaled features fits very well. Highest accuracy, F1, and MCC of all 5 models. |
| Decision Tree               | Weakest of the 5. A single unpruned tree overfits the training split and doesn't generalize as smoothly as the ensemble/linear methods — visible in the noticeably lower AUC. |
| kNN                         | Very strong, and achieved perfect recall (caught every malignant/benign case correctly on the positive class) — but that can come at the cost of borderline points near class boundaries being sensitive to the choice of *k* and scaling. |
| Naive Bayes                 | Solid AUC despite the independence assumption between features clearly not holding for correlated measurements like radius/perimeter/area — but its accuracy and precision trail the top models since that assumption still costs it at the decision boundary. |
| Random Forest (Ensemble)    | Consistently strong and stable — bagging many trees fixes most of the single Decision Tree's overfitting, giving the 2nd-highest AUC while staying robust without needing feature scaling. |
| **Overall Winner for your dataset?** | **Logistic Regression** — highest Accuracy, Precision, Recall, F1, and MCC. Random Forest is the runner-up and arguably the safer choice on unseen, noisier real-world data since it doesn't rely on a linear-separability assumption. |

## Project Structure

```
project-folder/
│-- app.py                  # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv           # held-out test split (features + target)
│-- model/
│   │-- train_models.py     # trains all 5 models, saves them + metrics.csv
│   │-- metrics.csv          # evaluation metrics for all 5 models
│   └-- saved_models/        # fitted model + scaler artifacts (.joblib)
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # optional — saved_models/ already included
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (include `model/saved_models/*.joblib` so the app
   doesn't need to retrain on startup).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New app** → select this repository → branch `main` → main file
   `app.py` → **Deploy**.
4. Once deployed, the app URL is your **Live Streamlit App Link**.

> **TODO:** `Live Streamlit App Link: https://<your-app>.streamlit.app`

## Streamlit App Features

- **Dataset upload (CSV):** Upload your own test CSV (must contain the 30
  feature columns; an optional `target` column enables full evaluation).
- **Model selection dropdown:** Switch between all 5 trained classifiers.
- **Evaluation metrics display:** Accuracy, AUC, Precision, Recall, F1, MCC
  computed live on whichever data is loaded.
- **Confusion matrix & classification report:** Rendered for the selected
  model, plus a side-by-side comparison table across all 5 models.
