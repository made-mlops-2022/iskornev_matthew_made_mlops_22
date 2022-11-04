import pickle
import click
import hydra
import pandas as pd
import logging
from sklearn.metrics import f1_score
from src.models import PATH_TO_MODEL
from src.data import PATH_TO_DATA
from src.features import MyTransformer
from hydra.core.config_store import ConfigStore
from src.entities import TrainPipelineCfg

logger = logging.getLogger("test_model")
# logger.setLevel(logging.INFO)
# formatter_stdout = logging.Formatter(
#             "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S")
# stream_handler = logging.StreamHandler()
# stream_handler.setLevel(logging.INFO)
# stream_handler.setFormatter(formatter_stdout)
# logger.addHandler(stream_handler)


def run_predict_pipeline(
        feature_test: pd.DataFrame, target_true: pd.Series, file_for_target: str, param: TrainPipelineCfg
):
    logger.info('Transform test features with our custom transformer')
    trans = MyTransformer(
        feature_test,
        param.features.categorical,
        param.features.numerical
    )
    trans.fit()
    feature_test_processed = trans.transform()
    logger.info("Loading our model")
    with open(PATH_TO_MODEL.joinpath('model.pkl'), 'rb') as file:
        model = pickle.load(file)

    logger.info("Running predict with model")  # {cfg.model.name}")
    target_predict = model.predict(feature_test_processed)
    # tmp_path = PATH_TO_DATA.joinpath('predicted_target')
    # with open(file_for_target, 'w') as f:
    with open(file_for_target, 'w') as f:
        for elem in target_predict:
            f.writelines(f'{elem}\r\n')
    logger.info(f"Predicted target was written to the {file_for_target}")
    logger.info(f"f1_score on test data = {f1_score(target_true, target_predict)}")


@click.command()
@click.option('--path_train_data', '-ptd',
              default=PATH_TO_DATA.joinpath('raw/features_test.csv'),
              help='Please enter path to file with your train data. Default path - data/raw/features_test.csv')
@click.option('--path_pred_target', '-ppt',
              default=PATH_TO_DATA.joinpath("predicted_target.txt"),
              help='Please enter path to file where you want save predicted target. Default path - '
                   'data/predicted_target.txt')
def main_wrapper(path_train_data: str, path_pred_target: str):
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=TrainPipelineCfg)

    @hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
    def _main(param: TrainPipelineCfg):
        logger.info(f"Data for test was taken from #{path_train_data}")
        feature_test = pd.read_csv(path_train_data)
        target_true = pd.read_csv(PATH_TO_DATA.joinpath('raw/target_test.csv'))
        run_predict_pipeline(feature_test, target_true, path_pred_target, param)
    _main()


if __name__ == '__main__':
    main_wrapper()
