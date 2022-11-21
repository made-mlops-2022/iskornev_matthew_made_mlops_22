import logging
from urllib.parse import urlparse
import pandas as pd
import mlflow
from random import shuffle
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import os

from src.features import MyTransformer
from src.entities import TrainPipelineCfg


os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
logger = logging.getLogger("train_model")


def train_model(
        X_train_processed: pd.DataFrame, X_train: pd.DataFrame, target: pd.Series, param: TrainPipelineCfg
) -> object:
    with mlflow.start_run():
        if param.model.name == 'knn':
            model = KNeighborsClassifier(n_neighbors=param.model.num_n)
            param_grid = {'n_neighbors': param.knn_grid.n_neighbors,
                          'metric': param.knn_grid.metric}
        else:
            model = RandomForestClassifier(n_estimators=param.model.n_estimators,
                                           max_depth=param.model.max_depth,
                                           random_state=param.model.random_state)
            param_grid = {'n_estimators': param.rfc_grid.n_estimators,
                          'max_depth': param.rfc_grid.max_depth,
                          'max_features': param.rfc_grid.max_features}
        if param.model.grid_search:
            logger.info("Use grid_search = True")
            logger.info(f"grid parameters = {param_grid}")
            length = len(X_train)
            train_size = 0.8
            len_train = round(length * train_size)
            arr_zero = [-1 for _ in range(len_train)]
            arr_m_one = [0 for _ in range(length - len_train)]
            val_split = arr_zero + arr_m_one
            shuffle(val_split)
            ps = PredefinedSplit(val_split)
            train_index, val_index = next(ps.split())
            df_train = X_train.iloc[train_index]
            df_val = X_train.iloc[val_index]
            trans = MyTransformer(
                param.features.categorical,
                param.features.numerical
            )
            trans.fit(df_train)
            df_train_processed = trans.transform(df_train)
            df_train_processed.set_index([pd.Index(train_index)], inplace=True)
            df_val_processed = trans.transform(df_val)
            df_val_processed.set_index([pd.Index(val_index)], inplace=True)
            X_for_val = pd.concat([df_train_processed, df_val_processed])
            X_for_val.sort_index(inplace=True)
            X_for_val.fillna(0, inplace=True)
            model_gs = GridSearchCV(estimator=model, param_grid=param_grid,
                                    scoring='f1',
                                    cv=PredefinedSplit(val_split))
            model_gs.fit(X_for_val, target)
            logger.info(f"best parameters = {model_gs.best_params_}")
            for key in model_gs.best_params_:
                mlflow.log_param(key, model_gs.best_params_[key])

            y_predict_train = model_gs.predict(X_train_processed)
            metrics = {'f1_score': f1_score(target, y_predict_train),
                       'accuracy_score': accuracy_score(target, y_predict_train),
                       'roc_auc_score': roc_auc_score(target, y_predict_train)}

            for key in metrics:
                mlflow.log_metric(key, metrics[key])

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    sk_model=model_gs,
                    artifact_path="classification_model",
                    registered_model_name=param.model.name)
            else:
                mlflow.sklearn.log_model(
                    sk_model=model_gs,
                    artifact_path="classification_model")

            return metrics, model_gs

        if param.model.name == 'knn':
            mlflow.log_param("n_neighbors", param.model.num_n)
            mlflow.log_param("metric", param.model.metric)
        else:
            mlflow.log_param("n_estimators", param.model.n_estimators)
            mlflow.log_param("max_depth", param.model.max_depth)
            mlflow.log_param("max_features", param.model.max_features)
        logger.info('Use grid_search = False')
        model.fit(X_train_processed, target)
        y_predict_train = model.predict(X_train_processed)
        metrics = {'f1_score': f1_score(target, y_predict_train),
                   'accuracy_score': accuracy_score(target, y_predict_train),
                   'roc_auc_score': roc_auc_score(target, y_predict_train)}
        for key in metrics:
            mlflow.log_metric(key, metrics[key])
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="classification_model",
                registered_model_name=param.model.name)
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="classification_model")
        return metrics, model
