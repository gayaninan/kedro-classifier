# Iris Classifier Pipeline

> *Note:* This project is an example of MLOps best practices using Kedro 1.x, demonstrating modular pipeline design and experiment tracking to follow key MLOps stages.

---

## 📌 Overview

This pipeline demonstrates a modular and reproducible machine learning workflow using the Iris dataset, following MLOps principles:

1. **Feature Engineering** – Create and transform features (e.g., feature interactions, normalisation).
2. **Data Processing** – Split the data into training and testing sets using parameters from `conf/base/parameters.yml`.
3. **Training** – Fit a logistic regression model using scikit-learn, logging parameters and metrics to MLflow and optionally Weights & Biases (W&B).
4. **Inference** – Run predictions on new or test data.
5. **Monitoring** – Perform data drift analysis and summary statistics between training and inference data.
6. **Retraining** – Retrain the model with new data if drift is detected or performance degrades.

---

## 🔁 Pipeline Inputs

### `iris_data`

| Field | Description |
|-------|-------------|
| Type  | `pandas.DataFrame` |
| Description | Raw iris data containing features and target column |

### `parameters`

| Field | Description |
|-------|-------------|
| Type  | `dict` |
| Description | Pipeline configuration parameters such as `train_fraction`, `random_state`, `target_column`, and `n_epochs` |

---

## 🔄 Intermediate Outputs

### `iris_data_fe`

| Field | Description |
|-------|-------------|
| Type  | `pandas.DataFrame` |
| Description | Feature-engineered iris data |

### `iris_data_norm`

| Field | Description |
|-------|-------------|
| Type  | `pandas.DataFrame` |
| Description | Normalised iris data |

### `X_train`, `X_test`

| Field | Description |
|-------|-------------|
| Type  | `pandas.DataFrame` |
| Description | Feature sets for training and inference |

### `y_train`, `y_test`

| Field | Description |
|-------|-------------|
| Type  | `pandas.DataFrame` |
| Description | Labels for training and evaluation |

---

## ✅ Final Outputs

### `model`

| Field | Description |
|-------|-------------|
| Type  | `sklearn.base.BaseEstimator` |
| Description | Trained model artifact |

### `y_pred`

| Field | Description |
|-------|-------------|
| Type  | `pandas.DataFrame` |
| Description | Predicted class labels from inference pipeline |

### `accuracy`

| Field | Description |
|-------|-------------|
| Type  | `float` |
| Description | Model accuracy on test set (optional evaluation step) |

### `drift_report`

| Field | Description |
|-------|-------------|
| Type  | `dict` |
| Description | Drift statistics comparing training vs inference inputs |

---

## 📊 Experiment Tracking

- **MLflow** – Logs parameters, metrics, and model artifacts.  
  Start with `mlflow ui` → [http://localhost:5000](http://localhost:5000)

- **Weights & Biases (optional)** – Logs visual metrics and artifacts.  
  View at [https://wandb.ai/](https://wandb.ai/) under your account/project.

---

## 🧩 Modular Pipelines

Each stage is defined in a dedicated module under `src/iris_classifier/pipelines/`:

- `feature_engineering`
- `data_processing`
- `training`
- `inference`
- `monitoring`
- `retraining`

---

## 🚀 How to Run

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the full pipeline**  
   ```bash
   kedro run
   ```

3. **Run individual pipelines**  
   ```bash
   kedro run --pipeline inference
   ```

4. **Track experiments**  
   ```bash
   mlflow ui
   ```

---

## 📝 Notes

- All code uses `pandas`, `numpy`, and `scikit-learn`.
- Kedro modular structure supports team collaboration and scaling.
- Inference logic is isolated and reusable across retraining or batch scoring.

---

## 📂 Outputs

| File                    | Description                   |
|-------------------------|-------------------------------|
| `model.pkl`             | Trained model artifact        |
| `y_pred.csv`            | Inference results             |
| `drift_report.csv`      | Tabular drift summary         |
| `drift_bar.png`         | Visual drift chart            |
| `model_partitions/`     | Timestamped outputs by run    |

