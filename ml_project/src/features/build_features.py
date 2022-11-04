import pandas as pd
from typing import List
# from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg',
#                         'exang', 'slope', 'ca', 'thal']
# NUMERIC_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, X: pd.DataFrame, categorical_features: List[str], numerical_features: List[str]):
        self.x = X
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        pass

    def process_categorical_features(self, categorical_df: pd.DataFrame) -> pd.DataFrame:
        categorical_pipeline = self.build_categorical_pipeline()
        categorical_pipeline.fit(categorical_df)
        col_list = categorical_pipeline.get_feature_names_out()
        df_tmp = pd.DataFrame(categorical_pipeline.transform(categorical_df).toarray(),
                              columns=col_list)
        num_pipeline = self.build_numerical_pipeline()
        res = pd.DataFrame(num_pipeline.fit_transform(df_tmp),
                           columns=col_list)
        return res

    @staticmethod
    def build_categorical_pipeline() -> Pipeline:
        categorical_pipeline = Pipeline(
            [
                # ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
                ("ohe", OneHotEncoder()),
            ]
        )
        return categorical_pipeline

    def process_numerical_features(self, numerical_df: pd.DataFrame) -> pd.DataFrame:
        num_pipeline = self.build_numerical_pipeline()
        return pd.DataFrame(num_pipeline.fit_transform(numerical_df),
                            columns=self.numerical_features)

    @staticmethod
    def build_numerical_pipeline() -> Pipeline:
        num_pipeline = Pipeline(
            [('scaler', StandardScaler())]
        )
        return num_pipeline

    def fit(self):
        return self

    def transform(self) -> pd.DataFrame:
        cat_x = self.x[self.categorical_features]
        cat_x_processed = self.process_categorical_features(cat_x)

        num_x = self.x[self.numerical_features]
        num_x_processed = self.process_numerical_features(num_x)
        res = pd.concat([num_x_processed, cat_x_processed], axis=1)
        return res
