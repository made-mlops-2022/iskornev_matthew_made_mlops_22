import os
import click
import pickle
import pandas as pd
from pathlib import Path

from my_transformer import MyTransformer


CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
NUMERICAL = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


@click.command("preprocess_data")
@click.option("--input_dir",
              default='../../data/splitted/raw/train/',
              help='Please enter path to input data. Default path - ../../data/raw/2022-11-28/')
@click.option("--output_dir",
              default='../../data/splitted/processed/train/',
              help='Please enter path to output data. Default path - ../../data/processed/')
@click.option("--val_dir",
              default='../../data/validation_artefacts/',
              help='Please enter path to validation artefacts. Default path - ../../data/validation_artefacts/')
def preprocess_data(input_dir: str, output_dir: str, val_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    data = pd.read_csv(Path(input_dir).joinpath('data.csv'))
    target = pd.read_csv(Path(input_dir).joinpath('target.csv'))

    trans = MyTransformer(
        CATEGORICAL,
        NUMERICAL
    )

    trans.fit(data)
    with open(Path(val_dir).joinpath('trans.pkl'), 'wb') as f:
        pickle.dump(trans, f)

    data_processed = trans.transform(data)

    data_processed.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == "__main__":
    preprocess_data()
