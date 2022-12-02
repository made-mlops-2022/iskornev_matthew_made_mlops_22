import os
import click
import pickle
import mlflow
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import numpy as np


os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5050"


@click.command("train")
@click.option("--input_dir",
              default='/data/splitted/processed/train/2022-11-28/',
              help='Please enter path to processed train data. Default path - '
                   '/data/splitted/processed/train/2022-11-28/')
@click.option("--output_dir",
              default='/data/validation_artefacts/',
              help='Please enter path to validation artefacts (for model and transformer). Default path - '
                   '/data/validation_artefacts/')
@click.option("--model_params", "-d", "model_params", type=(str, int), multiple=True,
              help="Enter parameters for model in format: -d 'n_estimatirs' number")
def train(input_dir: str, output_dir: str, model_params) -> None:
    with mlflow.start_run():
        model_params = dict(model_params)
        data = pd.read_csv(Path(input_dir).joinpath('train_data.csv'))
        target = pd.read_csv(Path(input_dir).joinpath('target.csv'))
        target = np.array(target).reshape(len(data))

        model = RandomForestClassifier(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            max_features='log2',
            random_state=model_params['random_state']
        )
        for key in model.get_params():
            mlflow.log_param(key, model.get_params()[key])

        model.fit(data, target)

        run_id = mlflow.active_run().info.run_id

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="classification_model",
            registered_model_name="model_rfc")

        mlflow.log_artifact(f"{output_dir}/trans.pkl")

        MODEL_NAME = f"model_rfc_{run_id}.pkl"
        with open(Path(output_dir).joinpath(MODEL_NAME), 'wb') as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    train()
