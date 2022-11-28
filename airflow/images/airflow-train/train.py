import os
import click
import pickle
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np


N_NEIGHBORS = 3
MODEL_NAME = 'model_rfc.pkl'


@click.command("train")
@click.option("--input_dir",
              default='../../data/splitted/processed/train/2022-11-28/',
              help='Please enter path to input data. Default path - ../../data/splitted/processed/train/2022-11-28/')
@click.option("--output_dir",
              default='../../data/validation_artefacts/',
              help='Please enter path to validation artefacts. Default path - ../../data/validation_artefacts/')
def train(input_dir: str, output_dir: str) -> None:
    data = pd.read_csv(Path(input_dir).joinpath('train_data.csv'))
    target = pd.read_csv(Path(input_dir).joinpath('target.csv'))
    target = np.array(target).reshape(len(data))

    # model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    model = RandomForestClassifier(n_estimators=150, max_depth=5, max_features='log2', random_state=42)
    # print(np.array(target).reshape(len(data)))
    # print(target)

    model.fit(data, target)

    # metrics = {'f1_score': f1_score(target, y_predict_train),
    #            'accuracy_score': accuracy_score(target, y_predict_train),
    #            'roc_auc_score': roc_auc_score(target, y_predict_train)}

    with open(Path(output_dir).joinpath(MODEL_NAME), 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train()
