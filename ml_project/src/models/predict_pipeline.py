import pickle
import click
import os
from pathlib import Path
import hydra
import pandas as pd
import logging
from sklearn.metrics import f1_score
from hydra.core.config_store import ConfigStore
from src.entities import TrainPipelineCfg

logger = logging.getLogger("test_model")


def run_predict_pipeline(
        feature_test: pd.DataFrame, target_true: pd.Series, file_for_target: str, param: TrainPipelineCfg
):
    logger.info('Transform test features with our custom transformer')
    with open(param.paths.path_to_transformer, 'rb') as file:
        trans = pickle.load(file)
    feature_test_processed = trans.transform(feature_test)
    logger.info("Loading our model")
    with open(param.paths.path_to_model, 'rb') as file:
        model = pickle.load(file)

    logger.info("Running predict with model")
    target_predict = model.predict(feature_test_processed)
    with open(file_for_target, 'w') as f:
        for elem in target_predict:
            f.writelines(f'{elem}\r\n')
    logger.info(f"Predicted target was written to the {file_for_target}")
    logger.info(f"f1_score on test data = {f1_score(target_true, target_predict)}")


@click.command()
@click.option('--path_train_data', '-ptd',
              default='data/raw/features_test.csv',
              help='Please enter path to file with your train data. Default path - data/raw/features_test.csv')
@click.option('--path_pred_target', '-ppt',
              default="data/predicted_target.txt",
              help='Please enter path to file where you want save predicted target. Default path - '
                   'data/predicted_target.txt')
def main_wrapper(path_train_data: str, path_pred_target: str):
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=TrainPipelineCfg)

    @hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
    def _main(param: TrainPipelineCfg):
        logger.info(f"Data for test was taken from #{path_train_data}")
        feature_test = pd.read_csv(path_train_data)
        tmp_path = os.path.dirname(param.paths.path_to_raw_data)
        target_true = pd.read_csv(Path(tmp_path).joinpath('target_test.csv'))
        run_predict_pipeline(feature_test, target_true, path_pred_target, param)
    _main()


if __name__ == '__main__':
    main_wrapper()
