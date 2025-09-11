import pandas as pd
import wandb
import mlflow

def run_inference(model_dict, X_new: pd.DataFrame) -> pd.DataFrame:
    clf = model_dict["model"]
    y_pred = clf.predict(X_new)
    pred_df = pd.DataFrame({"prediction": y_pred}, index=X_new.index)

    # Log predictions table to W&B
    wandb.init(project="kedro-classifier", name="inference_run", reinit=True)
    wandb.log({"predictions": wandb.Table(dataframe=pred_df)})
    wandb.finish()

    # Log predictions to MLflow
    pred_df.to_csv("y_pred.csv", index=False)
    mlflow.set_experiment("kedro-classifier")
    mlflow.log_artifact("y_pred.csv")

    return pred_df