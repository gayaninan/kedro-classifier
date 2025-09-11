# Iris Classifier Pipeline

> *Note:* This project is an example of MLOps best practices using Kedro 1.x, demonstrating modular pipeline design and experiment tracking to follow key MLOps stages.

## Overview

This pipeline demonstrates a modular and reproducible machine learning workflow, following MLOps principles:

1. **Feature Engineering:** Create and transform features (e.g., feature interactions, normalization).
2. **Data Processing:** Split the data into training and testing sets using a configurable ratio from `conf/base/parameters.yml`.
3. **Training:** Fit a logistic regression model using scikit-learn, logging parameters and metrics to MLflow and Weights & Biases (W&B).
4. **Reporting:** Report model accuracy and log performance curves.
5. **Monitoring:** Monitor data drift and statistics between train and test sets.
6. **Retraining:** Optionally retrain the model with new data.

## Pipeline Inputs

### `iris_data`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Raw iris data containing features and target column |

### `parameters`

|      |                    |
| ---- | ------------------ |
| Type | `dict` |
| Description | Project parameters: `train_fraction`, `random_state`, `target_column`, `n_epochs` |

## Pipeline Intermediate Outputs

### `iris_data_fe`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Feature-engineered iris data |

### `iris_data_norm`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Normalised iris data |

### `X_train`, `X_test`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Train/test set features |

### `y_train`, `y_test`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Train/test set targets |

### `model`

|      |                    |
| ---- | ------------------ |
| Type | `sklearn.base.BaseEstimator` |
| Description | Trained scikit-learn model |

### `y_pred`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Model predictions on test set |

### `accuracy`

|      |                    |
| ---- | ------------------ |
| Type | `float` |
| Description | Model accuracy on test set |

### `drift_report`

|      |                    |
| ---- | ------------------ |
| Type | `dict` |
| Description | Data drift statistics between train and test sets |

## Experiment Tracking

- **MLflow:** Tracks parameters, metrics, and model artifacts.  
  Start the UI with `mlflow ui` and visit [http://localhost:5000](http://localhost:5000).
- **Weights & Biases (W&B):** Tracks parameters, metrics, curves, and model/data artifacts.  
  View your runs at [https://wandb.ai/](https://wandb.ai/) under your project.

## Modular Pipelines

- `feature_engineering`
- `data_processing`
- `training`
- `reporting`
- `monitoring`
- `retraining`

Each pipeline is defined in its own folder under `src/iris_classifier/pipelines/`.

## How to Run

1. Install dependencies:  
   `pip install -r requirements.txt`
2. Run the pipeline:  
   `kedro run`
3. View experiment tracking dashboards in MLflow and W&B.

## Notes

- All code uses pandas and numpy.
- Modular structure demonstrates best practices for scalable ML projects.
- Easily extendable for more complex models and data.
