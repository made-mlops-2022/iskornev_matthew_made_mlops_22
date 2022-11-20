import pandas as pd
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features: List[str], numerical_features: List[str]):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.encoder = None
        self.scaler = None
        pass

    def fit_categorical(self, categorical_df) -> Pipeline:
        categorical_pipeline = self.build_categorical_pipeline()
        categorical_pipeline.fit(categorical_df)
        return categorical_pipeline

    def fit_numerical(self, df: pd.DataFrame) -> Pipeline:
        num_pipeline = self.build_numerical_pipeline()
        num_pipeline.fit(df)
        return num_pipeline

    def fit(self, x_df: pd.DataFrame):
        cat_x = x_df[self.categorical_features]
        self.encoder = self.fit_categorical(cat_x)
        df_tmp_all = self.process_categorical_features(x_df, cat_x)
        self.scaler = self.fit_numerical(df_tmp_all)
        return self

    def process_categorical_features(self, x_df: pd.DataFrame, categorical_df: pd.DataFrame) -> pd.DataFrame:
        col_list = self.encoder.get_feature_names_out()
        df_cat_trans = pd.DataFrame(self.encoder.transform(categorical_df).toarray(),
                                    columns=col_list, index=categorical_df.index)
        res = pd.concat([x_df[self.numerical_features], df_cat_trans], axis=1)
        return res

    @staticmethod
    def build_categorical_pipeline() -> Pipeline:
        categorical_pipeline = Pipeline(
            [
                ("ohe", OneHotEncoder()),
            ]
        )
        return categorical_pipeline

    def process_numerical_features(self, df: pd.DataFrame, columns) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.transform(df), columns=columns)

    @staticmethod
    def build_numerical_pipeline() -> Pipeline:
        num_pipeline = Pipeline(
            [
                ('scaler', StandardScaler())
            ]
        )
        return num_pipeline

    def transform(self, x_df: pd.DataFrame) -> pd.DataFrame:
        cat_x = x_df[self.categorical_features]
        cat_x_processed = self.process_categorical_features(x_df, cat_x)
        columns = cat_x_processed.columns
        res = self.process_numerical_features(cat_x_processed, columns)
        return res
