import pandas as pd
import wandb
import mlflow
import matplotlib.pyplot as plt

def monitor_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    train_mean = X_train.mean().to_dict()
    test_mean = X_test.mean().to_dict()
    drift = {col: abs(train_mean[col] - test_mean[col]) for col in train_mean}

    # Convert drift to DataFrame for logging
    drift_df = pd.DataFrame(list(drift.items()), columns=["feature", "drift"])

    # Log drift table to W&B
    wandb.init(project="kedro-classifier", name="monitoring_run", reinit=True)
    wandb.log({"drift_report": wandb.Table(dataframe=drift_df)})

    # Log drift table to MLflow
    drift_df.to_csv("drift_report.csv", index=False)
    mlflow.set_experiment("kedro-classifier")
    mlflow.start_run()
    mlflow.log_artifact("drift_report.csv")

    # Visualise drift as a bar chart
    plt.figure(figsize=(8, 4))
    plt.bar(drift_df["feature"], drift_df["drift"])
    plt.xticks(rotation=45)
    plt.title("Feature Drift (Train vs Test)")
    plt.tight_layout()
    plt.savefig("drift_bar.png")
    wandb.log({"drift_bar": wandb.Image("drift_bar.png")})
    wandb.finish()
    mlflow.log_artifact("drift_bar.png")
    plt.close()

    return drift