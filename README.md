# 🫀 Stroke Risk Prediction System

A full-stack ML web app that predicts stroke risk from patient health data — built for both individual patients and doctors managing multiple patients at once.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## What this actually does

Stroke is one of the leading causes of death globally, and most prediction tools are either inaccessible to regular people or stuck inside hospital software. I wanted to build something a patient could open in a browser and actually use.

This system takes 10 health parameters, runs them through three ML models, combines their predictions using ensemble voting, and returns not just a risk score — but a risk *category* (Very Low → Very High) with the top contributing factors and specific health recommendations.

There are two separate interfaces built into the same app:
- **Patient view** — fill in your details, get your risk assessment instantly
- **Doctor dashboard** — upload a CSV of multiple patients, get batch predictions for all of them at once

---

## How the ML works

### The dataset

`healthcare-dataset-stroke-data.csv` — a real healthcare dataset with patient records including demographics, lifestyle factors, and medical history. Stroke events are the minority class (heavily imbalanced), which is one of the more interesting problems this project had to handle.

### Three models, trained separately

I trained three classifiers and saved each one independently:

| Model | File | Why it's included |
|---|---|---|
| Logistic Regression | `model_lr.pkl` | Fast, interpretable baseline — good for understanding which features have linear relationships with stroke risk |
| Decision Tree | `model_dt.pkl` | Captures non-linear interactions, easy to visualize which splits matter |
| Random Forest | `model_rf.pkl` | Reduces the overfitting problem the single Decision Tree has — 100 trees averaged together |

### Ensemble voting

The final prediction isn't from any single model — it's a **voting ensemble** (`model_ensemble.pkl`). All three models vote, and the majority wins. This is a classic technique for reducing variance: if one model is wrong on an edge case, the other two can override it.

### Preprocessing pipeline

Raw user input goes through the same pipeline the training data did:
- Categorical encoding (gender, work type, residence type, smoking status, marital status) saved in `model_columns.pkl` — so the feature order at inference time exactly matches training
- Numerical scaling via `scaler.pkl` (fitted StandardScaler) — age, BMI, average glucose level

This is actually the part that breaks most beginner ML projects: the model was trained on scaled features, so if you feed raw numbers at inference time, your predictions are garbage. Saving and reloading the scaler was a deliberate decision.

### Risk categorisation

The ensemble doesn't just output 0 or 1. The raw probability is mapped to five risk levels:

```
Very Low  → < 20%
Low       → 20–40%
Medium    → 40–60%
High      → 60–80%
Very High → > 80%
```

Each level comes with specific health recommendations — not just a number on a screen.

---

## Input Parameters

```
Age · Gender · Hypertension · Heart Disease
Average Glucose Level · BMI · Marital Status
Work Type · Residence Type · Smoking Status
```

---

## Project Structure

```
Stroke-Risk-Predictor/
│
├── app.py                          ← Flask app — routes, preprocessing, prediction logic
├── project_stroke.ipynb            ← Full ML pipeline: EDA → training → evaluation → export
│
├── model_lr.pkl                    ← Trained Logistic Regression
├── model_dt.pkl                    ← Trained Decision Tree
├── model_rf.pkl                    ← Trained Random Forest
├── model_ensemble.pkl              ← Voting ensemble (all three combined)
├── scaler.pkl                      ← Fitted StandardScaler (must match training)
├── model_columns.pkl               ← Feature column order (ensures consistent encoding)
│
├── healthcare-dataset-stroke-data.csv  ← Original dataset
│
├── templates/                      ← HTML pages (patient form, doctor dashboard, results)
├── static/                         ← CSS, any frontend assets
│
├── run_website.bat                 ← Windows one-click launcher
└── open_notebook.bat               ← Windows one-click notebook opener
```

---

## Running it locally

```bash
git clone https://github.com/harleenkhanuja/Stroke-Risk-Predictor.git
cd Stroke-Risk-Predictor
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

> **Windows users:** just double-click `run_website.bat` — it handles everything.

The models are pre-trained and committed as `.pkl` files, so you don't need to retrain anything. Open the app and it's ready.

---

## Notebook walkthrough (`project_stroke.ipynb`)

The Jupyter notebook covers the complete ML pipeline:

1. **Exploratory Data Analysis** — class imbalance visualization, feature distributions, correlation heatmap
2. **Preprocessing** — handling missing values (BMI has nulls), encoding categoricals, scaling numericals
3. **Model training** — Logistic Regression, Decision Tree, Random Forest — each trained and evaluated separately
4. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrices per model
5. **Ensemble construction** — combining all three via VotingClassifier
6. **Model export** — saving all `.pkl` files with `joblib` for use in Flask

---

## Why three models instead of just the best one

This is something I thought about during the project. A single Random Forest would probably have the best raw accuracy. But in a healthcare context, I wanted to understand *why* predictions are being made, not just what they are:

- Logistic Regression tells you the direct weight each feature has on the outcome — interpretable and fast
- Decision Tree shows you the exact decision path — which conditions, in which order, led to the risk level
- Random Forest gives the most reliable probability estimate

Running all three and combining them meant I could cross-check predictions. If all three agree it's high risk, that's a stronger signal than one model alone. The ensemble result is more trustworthy than any individual model, especially on the edge cases near the decision boundary.

---

## What I'd add next

- [ ] SMOTE or class weighting to better handle the stroke/no-stroke imbalance in the dataset
- [ ] XGBoost or LightGBM as a fourth model in the ensemble
- [ ] User authentication so patients can track their risk over time
- [ ] Deploy on Render so it's accessible without a local setup
- [ ] Explainability layer using SHAP values to show exactly which features drove each individual prediction

---

## Why I built this

Most ML projects stop at the notebook. You train a model, look at the accuracy, and call it done. I wanted to go the full distance — take that model and put it behind a real web interface that someone could actually open and use. The doctor dashboard came from thinking: *what would actually be useful to a real user?* A clinician doesn't want to enter one patient at a time. They'd want to upload a file and see results for all of them.

That's the gap this project tries to close.

---

**Made by [Harleen Khanuja](https://www.linkedin.com/in/harleenkhanuja/)**  
