import pandas as pd
import mlflow
import wandb
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score
from typing import Dict

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict) -> Dict:
    mlflow.set_experiment("kedro-classifier")
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.start_run()
    mlflow.log_params(parameters)

    wandb.init(project="kedro-classifier", name="training_run", reinit=True)
    wandb.config.update(parameters)


    y_train_series = y_train.iloc[:, 0]
    n_epochs = parameters.get("n_epochs", 10)
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True, random_state=42)

    losses = []
    accuracies = []

    for epoch in range(n_epochs):
        clf.fit(X_train, y_train_series)
        y_pred_proba = clf.predict_proba(X_train)
        y_pred = clf.predict(X_train)
        loss = log_loss(y_train_series, y_pred_proba)
        acc = accuracy_score(y_train_series, y_pred)
        losses.append(loss)
        accuracies.append(acc)

        wandb.log({"epoch": epoch, "loss": loss, "accuracy": acc})
        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("accuracy", acc, step=epoch)

    wandb.log({"loss_curve": wandb.plot.line_series(xs=list(range(n_epochs)), ys=[losses], keys=["loss"], title="Loss Curve")})
    wandb.log({"accuracy_curve": wandb.plot.line_series(xs=list(range(n_epochs)), ys=[accuracies], keys=["accuracy"], title="Accuracy Curve")})

    mlflow.sklearn.log_model(clf, "model")
    wandb.finish()
    mlflow.end_run()

    return {"model": clf}