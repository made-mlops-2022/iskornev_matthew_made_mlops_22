from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
import unittest
from sklearn.model_selection import train_test_split
from src.features import MyTransformer
from src.entities.train_params import SplittingParams, FeatureList, ModelParams, TrainPipelineCfg
from src.models import train_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = pd.read_csv('tests/synth_data/synthetic_data.csv')
        self.X = self.data.drop(columns=['condition'])
        self.target = self.data['condition']
        self.data_shape = (100, 14)
        self.split_param = SplittingParams()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.target,
                             test_size=self.split_param.test_size,
                             random_state=self.split_param.random_state)
        self.categorical = ['sex', 'cp', 'fbs', 'restecg',
                            'exang', 'slope', 'ca', 'thal']
        self.numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        self.feat_param = FeatureList(
            categorical=self.categorical,
            numerical=self.numerical
        )
        trans_train = MyTransformer(
            self.X_train,
            self.categorical,
            self.numerical
        )
        trans_train.fit()
        self.X_train_processed = trans_train.transform()
        trans_test = MyTransformer(
            self.X_test,
            self.categorical,
            self.numerical
        )
        trans_test.fit()
        self.X_test_processed = trans_test.transform()
        tpc = TrainPipelineCfg(
            model=ModelParams(),
            splitting_params=self.split_param,
            features=self.feat_param
        )
        self.metrics, self.model = train_model(self.X_train_processed, self.X_train, self.y_train, tpc)

    def test_transformer(self):
        expect = (len(self.X) * (1 - self.split_param.test_size), 13)
        expect_processed = (len(self.X) * (1 - self.split_param.test_size), 28)
        self.assertEqual(self.X_train.shape, expect)
        self.assertEqual(self.X_train_processed.shape, expect_processed)
        expect = (len(self.X) * self.split_param.test_size, 13)
        expect_processed = (len(self.X) * self.split_param.test_size, 28)
        self.assertEqual(self.X_test.shape, expect)
        self.assertEqual(self.X_test_processed.shape, expect_processed)
        self.assertIsInstance(self.X_train_processed, pd.DataFrame)
        self.assertIsInstance(self.X_test_processed, pd.DataFrame)

    def test_fit_model(self):
        self.assertIsInstance(self.model, KNeighborsClassifier)
        self.assertEqual(len(self.y_train), len(self.X) * (1 - self.split_param.test_size))
        tmp = list(np.unique(self.y_train))
        self.assertEqual(tmp, [0, 1])
        self.assertGreaterEqual(self.metrics['f1_score'], 0.7)
        self.assertGreaterEqual(self.metrics['accuracy_score'], 0.7)
        self.assertGreaterEqual(self.metrics['roc_auc_score'], 0.7)
        check_is_fitted(self.model)

    def test_predict_model(self):
        y_predicted = self.model.predict(self.X_test_processed)
        self.assertIsInstance(y_predicted, np.ndarray)
        self.assertEqual(len(y_predicted), len(self.X) * self.split_param.test_size)
        self.assertGreaterEqual(f1_score(self.y_test, y_predicted), 0.5)
        tmp = list(np.unique(self.y_train))
        self.assertEqual(tmp, [0, 1])

    def test_grid_search(self):
        custom_model_param = ModelParams(
            grid_search=True
        )
        tpc = TrainPipelineCfg(
            model=custom_model_param,
            splitting_params=self.split_param,
            features=self.feat_param
        )
        metrics, model_gs = train_model(self.X_train_processed, self.X_train, self.y_train, tpc)
        y_predict_train = model_gs.predict(self.X_train_processed)
        self.assertIsInstance(model_gs.best_params_, dict)
        self.assertIsInstance(self.model, KNeighborsClassifier)
        self.assertGreaterEqual(f1_score(self.y_train, y_predict_train), 0.5)
        self.assertEqual(len(self.y_train), len(self.X) * (1 - self.split_param.test_size))
        tmp = list(np.unique(self.y_train))
        self.assertEqual(tmp, [0, 1])
        self.assertGreaterEqual(metrics['f1_score'], 0.5)
        self.assertGreaterEqual(metrics['accuracy_score'], 0.5)
        self.assertGreaterEqual(metrics['roc_auc_score'], 0.5)
        check_is_fitted(self.model)
