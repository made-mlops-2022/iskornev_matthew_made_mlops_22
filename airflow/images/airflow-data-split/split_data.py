import os
import click
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


@click.command("split_data")
@click.option("--input_dir",
              default='/data/raw/2022-11-28',
              help='Please enter path to input data. Default path - /data/raw/2022-11-28')
@click.option("--output_train_dir",
              default='/data/splitted/raw/train/',
              help='Please enter path to output splitted train data. Default path - /data/splitted/raw/train/')
@click.option("--output_val_dir",
              default='/data/splitted/raw/val/',
              help='Please enter path to splitted validation data. Default path - /data/splitted/raw/val/')
@click.option("--train_size",
              type=float,
              help='Please enter size of train data')
def split_data(input_dir: str, output_train_dir: str, output_val_dir: str, train_size: float) -> None:
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    data = pd.read_csv(Path(input_dir).joinpath('data.csv'))
    target = pd.read_csv(Path(input_dir).joinpath('target.csv'))

    x_train, x_val, y_train, y_val = train_test_split(data, target,
                                                      train_size=train_size,
                                                      random_state=42)

    x_train.to_csv(os.path.join(output_train_dir, "data.csv"), index=False)
    y_train.to_csv(os.path.join(output_train_dir, "target.csv"), index=False)

    x_val.to_csv(os.path.join(output_val_dir, "data.csv"), index=False)
    y_val.to_csv(os.path.join(output_val_dir, "target.csv"), index=False)


if __name__ == "__main__":
    split_data()
