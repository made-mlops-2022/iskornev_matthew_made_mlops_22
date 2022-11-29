import click
import os
from os import walk
import pickle
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np
import mlflow


os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5050"
TRANSFORMER_PATH = 'mlflow_runs/0/'


@click.command("validation")
@click.option("--input_dir",
              default='../../data/splitted/raw/val/2022-11-28/',
              help='Please enter path to input data. Default path - ../../data/splitted/raw/val/2022-11-28/')
@click.option("--model_dir",
              default='../../data/validation_artefacts/2022-11-29/',
              help='Please enter path to validation artefacts. Default path - '
                   '../../data/validation_artefacts/2022-11-29/')
def validation(input_dir: str, model_dir: str) -> None:
    filenames = next(walk(model_dir), (None, None, []))[2]
    print(filenames)

    max_len = 0
    for filename in filenames:
        if len(filename) > max_len:
            max_len = len(filename)

    MODEL_NAME = ""
    for filename in filenames:
        if len(filename) == max_len:
            MODEL_NAME = filename
    print(MODEL_NAME)

    curr_run_id = MODEL_NAME[10: -4]
    print(curr_run_id)

    with mlflow.start_run(run_id=curr_run_id):
        data = pd.read_csv(Path(input_dir).joinpath('data.csv'))
        target = pd.read_csv(Path(input_dir).joinpath('target.csv'))
        target = np.array(target).reshape(len(data))

        with open(f"{TRANSFORMER_PATH}/{curr_run_id}/artifacts/trans.pkl", 'rb') as f:
            trans = pickle.load(f)

        data = trans.transform(data)

        with open(Path(model_dir).joinpath(MODEL_NAME), 'rb') as f:
            model = pickle.load(f)

        y_predict = model.predict(data)

        metrics = {'f1_score': f1_score(target, y_predict),
                   'accuracy_score': accuracy_score(target, y_predict),
                   'roc_auc_score': roc_auc_score(target, y_predict)}

        for key in metrics:
            mlflow.log_metric(key, metrics[key])

        with open(Path(model_dir).joinpath('metrics.json'), 'w') as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    validation()
