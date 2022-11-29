import click
import os
import pickle
import pandas as pd
from pathlib import Path


MODEL_NAME = 'model_rfc.pkl'
TRANSFORMER_NAME = 'trans.pkl'


@click.command("predict")
@click.option("--input_dir",
              default='../../data/raw/2022-11-29/',
              help='Please enter path to input data. Default path - ../../data/raw/2022-11-29/')
@click.option("--output_dir",
              default='../../data/predictions/',
              help='Please enter path to output dur. Default path - '
                   '../../data/predictions/')
@click.option("--model_dir",
              default='../../data/validation_artefacts/2022-11-28/',
              help='Please enter path to current model and transformer. Default path - '
                   '../../data/validation_artefacts/2022-11-28/')
def predict(input_dir: str, output_dir: str, model_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(Path(input_dir).joinpath('data.csv'))

    with open(Path(model_dir).joinpath(TRANSFORMER_NAME), 'rb') as f:
        trans = pickle.load(f)

    data = trans.transform(data)

    with open(Path(model_dir).joinpath(MODEL_NAME), 'rb') as f:
        model = pickle.load(f)

    y_predict = model.predict(data)

    df_pred = pd.DataFrame(y_predict, columns=['target'])
    df_pred.to_csv(Path(output_dir).joinpath('predictions.csv'), index=False)


if __name__ == "__main__":
    predict()
