import click
import os
import pickle
import pandas as pd
from pathlib import Path
import mlflow


os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5050"
TRANSFORMER_PATH = 'mlflow_runs/0/'


@click.command("predict")
@click.option("--input_dir",
              default='data/raw/2022-11-29/',
              help='Please enter path to input data. Default path - ../../data/raw/2022-11-29/')
@click.option("--output_dir",
              default='data/predictions/',
              help='Please enter path to output dur. Default path - '
                   '../../data/predictions/')
@click.option("--model_dir",
              default='data/validation_artefacts/2022-11-28/',
              help='Please enter path to current model and transformer. Default path - '
                   '../../data/validation_artefacts/2022-11-28/')
def predict(input_dir: str, output_dir: str, model_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(Path(input_dir).joinpath('data.csv'))

    client = mlflow.MlflowClient()
    model_run_id = ""
    for rm in client.search_registered_models():
        rm = dict(rm)
        if rm['name'] == 'model_rfc':
            for elem in rm['latest_versions']:
                elem = dict(elem)
                if elem['current_stage'] == 'Production':
                    model_run_id = elem['run_id']
                    print(model_run_id)

    stage = 'Production'
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/model_rfc/{stage}"
    )

    # dict_model_info = client.search_registered_models()
    # print(dict_model_info)

    with open(f"{TRANSFORMER_PATH}/{model_run_id}/artifacts/trans.pkl", 'rb') as f:
        trans = pickle.load(f)

    data = trans.transform(data)

    y_predict = model.predict(data)

    df_pred = pd.DataFrame(y_predict, columns=['target'])
    df_pred.to_csv(Path(output_dir).joinpath('predictions.csv'), index=False)


if __name__ == "__main__":
    predict()
