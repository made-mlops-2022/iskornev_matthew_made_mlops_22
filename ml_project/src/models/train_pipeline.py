import hydra
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models import train_model, PATH_TO_MODEL
from src.data import PATH_TO_DATA
from src.features import MyTransformer
from hydra.core.config_store import ConfigStore
from src.entities import TrainPipelineCfg


# from entities import TrainConfig
# logger.setLevel(logging.INFO)
# formatter_stdout = logging.Formatter(
#             "%(asctime)s\t%(levelname)s\t%(name)20s\t%(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S")
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter_stdout)
# logger.addHandler(stream_handler)
# logger.info(f"f1_score on train data = ")
# logger.info("ssss")
logger = logging.getLogger("train_model")


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def run_train_pipeline(param: TrainPipelineCfg):
    logger.info(f'Start train with model {param.model.name}')
    df = pd.read_csv(PATH_TO_DATA.joinpath('raw/heart_cleveland_upload.csv'))
    X = df.drop(columns=['condition'])
    target = df['condition']
    logger.info(f'Split our data with test_size {param.splitting_params.test_size}')
    X_train, X_test, y_train, y_test = train_test_split(X, target,
                                                        test_size=param.splitting_params.test_size,
                                                        random_state=param.splitting_params.random_state)
    logger.info('Transform train features with our custom transformer')
    trans = MyTransformer(
        X_train,
        param.features.categorical,
        param.features.numerical
    )
    trans.fit()
    X_train_processed = trans.transform()
    y_train.reset_index(inplace=True, drop=True)
    df_processed = pd.concat([X_train_processed, pd.DataFrame(y_train, columns=['condition'])], axis=1)
    logger.info('Load process data to "data/processed/heart_cleveland_upload.csv"')
    df_processed.to_csv(PATH_TO_DATA.joinpath('processed/heart_cleveland_upload.csv'), index=False)

    logger.info('Load unprocessed test features to "data/raw/feature_test.csv"')
    logger.info('Load unprocessed test target to "data/raw/target_test.csv"')
    tmp_path = PATH_TO_DATA.joinpath('raw')
    X_test.to_csv(tmp_path.joinpath('features_test.csv'), index=False)
    y_test.to_csv(tmp_path.joinpath('target_test.csv'), index=False)
    logger.info('Training model')
    metrics, model = train_model(X_train_processed, X_train, y_train, param)
    logger.info(f"{param.model.name} f1_score on train data = {metrics['f1_score']}")
    with open(PATH_TO_MODEL.joinpath('model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    logger.info('Model was load to data/model/model.pkl')


def main():
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=TrainPipelineCfg)
    run_train_pipeline()


if __name__ == '__main__':
    main()
