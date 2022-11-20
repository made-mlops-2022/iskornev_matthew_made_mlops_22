import hydra
import pickle
from pathlib import Path
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models import train_model
from src.features import MyTransformer
from hydra.core.config_store import ConfigStore
from src.entities import TrainPipelineCfg


logger = logging.getLogger("train_model")


def get_data_with_target(dataframe: pd.DataFrame, target: str):
    X = dataframe.drop(columns=target)
    target = dataframe[target]
    return X, target


@hydra.main(version_base=None, config_path='../config', config_name='config.yaml')
def run_train_pipeline(param: TrainPipelineCfg):
    logger.info(f'Start train with model {param.model.name}')
    df = pd.read_csv(param.paths.path_to_raw_data)
    X, target = get_data_with_target(df, param.features.target)
    logger.info(f'Split our data with test_size {param.splitting_params.test_size}')
    X_train, X_test, y_train, y_test = train_test_split(X, target,
                                                        test_size=param.splitting_params.test_size,
                                                        random_state=param.splitting_params.random_state)
    logger.info('Transform train features with our custom transformer')
    trans = MyTransformer(
        param.features.categorical,
        param.features.numerical
    )
    trans.fit(X_train)
    with open(param.paths.path_to_transformer, 'wb') as f:
        pickle.dump(trans, f)
    X_train_processed = trans.transform(X_train)
    y_train.reset_index(inplace=True, drop=True)
    df_processed = pd.concat([X_train_processed, pd.DataFrame(y_train, columns=[param.features.target])], axis=1)
    logger.info('Load process data to "data/processed/heart_cleveland_upload.csv"')
    df_processed.to_csv(param.paths.path_to_processed_data, index=False)

    logger.info('Load unprocessed test features to "data/raw/feature_test.csv"')
    logger.info('Load unprocessed test target to "data/raw/target_test.csv"')
    tmp_path = os.path.dirname(param.paths.path_to_raw_data)
    X_test.to_csv(Path(tmp_path).joinpath('features_test.csv'), index=False)
    y_test.to_csv(Path(tmp_path).joinpath('target_test.csv'), index=False)
    logger.info('Training model')
    metrics, model = train_model(X_train_processed, X_train, y_train, param)
    logger.info(f"{param.model.name} f1_score on train data = {metrics['f1_score']}")
    with open(param.paths.path_to_model, 'wb') as f:
        pickle.dump(model, f)
    logger.info('Model was load to data/model/model.pkl')


def main():
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=TrainPipelineCfg)
    run_train_pipeline()


if __name__ == '__main__':
    main()
