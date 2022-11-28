import click
import pickle
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np


MODEL_NAME = 'model_rfc.pkl'


@click.command("validation")
@click.option("--input_dir",
              default='../../data/splitted/raw/val/2022-11-28/',
              help='Please enter path to input data. Default path - ../../data/splitted/raw/val/2022-11-28/')
@click.option("--model_dir",
              default='../../data/validation_artefacts/2022-11-28/',
              help='Please enter path to validation artefacts. Default path - '
                   '../../data/validation_artefacts/2022-11-28/')
def validation(input_dir: str, model_dir: str) -> None:
    data = pd.read_csv(Path(input_dir).joinpath('data.csv'))
    target = pd.read_csv(Path(input_dir).joinpath('target.csv'))
    target = np.array(target).reshape(len(data))

    with open(Path(model_dir).joinpath('trans.pkl'), 'rb') as f:
        trans = pickle.load(f)

    data = trans.transform(data)

    with open(Path(model_dir).joinpath(MODEL_NAME), 'rb') as f:
        model = pickle.load(f)

    y_predict = model.predict(data)

    metrics = {'f1_score': f1_score(target, y_predict),
               'accuracy_score': accuracy_score(target, y_predict),
               'roc_auc_score': roc_auc_score(target, y_predict)}

    with open(Path(model_dir).joinpath('metrics.json'), 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    validation()
