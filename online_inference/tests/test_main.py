import pytest
from fastapi.testclient import TestClient
import json
import os

from microservice.main import app
from microservice.main import load_model
from microservice.entities import TRUE_ORDER_FEATURES


client = TestClient(app)

os.environ.setdefault('URL', 'http://127.0.0.1:8000/predict')

data = {
    'age': 23,
    'sex': 1,
    'cp': 0,
    'trestbps': 120,
    'chol': 100,
    'fbs': 0,
    'restecg': 0,
    'thalach': 100,
    'exang': 0,
    'oldpeak': 1,
    'slope': 0,
    'ca': 0,
    'thal': 0
}


@pytest.fixture(scope='session', autouse=True)
def initialize_model():
    load_model()


def test_correct_no_disease():
    response = client.post(os.getenv('URL'), content=json.dumps(data))
    assert response.status_code == 200
    assert response.json() == {'label': 'No disease', 'prediction': 0, 'probability': 0.72}
    assert response.json()['label'] == 'No disease'
    assert response.json()['prediction'] == 0
    assert response.json()['probability'] == 0.72


def test_correct_disease():
    data_tmp = data.copy()
    data_tmp['age'] = 67
    data_tmp['cp'] = 3
    data_tmp['trestbps'] = 180
    data_tmp['chol'] = 600
    data_tmp['fbs'] = 1
    data_tmp['slope'] = 2
    data_tmp['thal'] = 2
    response = client.post('/predict', content=json.dumps(data_tmp))
    assert response.status_code == 200
    assert response.json()['label'] == 'Disease'
    assert response.json()['prediction'] == 1
    assert response.json()['probability'] == 0.61


def test_validation():
    def testing_int_field(feat_name, val, msg=None):
        data_tmp = data.copy()
        data_tmp[feat_name] = val
        response = client.post('/predict', content=json.dumps(data_tmp))
        assert response.status_code == 400
        if msg:
            assert response.json()['detail'] == msg

    val = -10
    msg = f"You print incorrect age {val}."
    testing_int_field("age", val, msg)
    testing_int_field("age", 200)

    val = 3
    msg = f"You print incorrect sex: {val}. Please input (1 = male; 0 = female)."
    testing_int_field("sex", val, msg)
    testing_int_field("sex", -1)

    testing_int_field("cp", -1.5)
    testing_int_field("cp", 4)

    testing_int_field("trestbps", 300)
    testing_int_field("trestbps", 0)

    testing_int_field("chol", 700)
    testing_int_field("chol", -100)

    testing_int_field("fbs", -5)
    testing_int_field("fbs", 2)

    testing_int_field("restecg", -5)
    testing_int_field("restecg", 3)

    val = -3
    msg = f"You print incorrect maximum heart rate achieved {val}."
    testing_int_field("thalach", val, msg)
    val = 300
    msg = f"You print incorrect maximum heart rate achieved {val}."
    testing_int_field("thalach", val, msg)

    testing_int_field("exang", -1)
    testing_int_field("exang", 3)

    val = -1.0
    msg = f"You print incorrect ST depression induced by exercise relative to rest {val}."
    testing_int_field("oldpeak", val, msg)
    val = 8.0
    msg = f"You print incorrect ST depression induced by exercise relative to rest {val}."
    testing_int_field("oldpeak", val, msg)

    testing_int_field("slope", -1)
    testing_int_field("slope", 3)

    testing_int_field("ca", -1)
    testing_int_field("ca", 5)

    testing_int_field("thal", -1)
    testing_int_field("thal", 3)


def test_missing_required_field():
    data_tmp = {
        'sex': 1,
        'cp': 0,
        'trestbps': 120,
        'chol': 100,
        'fbs': 0,
        'restecg': 0,
        'thalach': 100,
        'exang': 0,
        'oldpeak': 1,
        'slope': 0,
        'ca': 0,
        'thal': 0
    }
    response = client.post('/predict', content=json.dumps(data_tmp))
    true_response = f"You did not print feature age. List of features {TRUE_ORDER_FEATURES}."
    assert response.status_code == 400
    assert response.json()['detail'] == true_response


def test_order_fields():
    data_tmp = {
        'ca': 0,
        'sex': 1,
        'cp': 0,
        'trestbps': 120,
        'chol': 100,
        'fbs': 0,
        'restecg': 0,
        'age': 33,
        'thalach': 100,
        'exang': 0,
        'oldpeak': 1,
        'slope': 0,
        'thal': 0
    }
    response = client.post('/predict', content=json.dumps(data_tmp))
    true_response = f"You print features in wrong order or with mistakes. Correct order {TRUE_ORDER_FEATURES}."
    assert response.status_code == 400
    assert response.json()['detail'] == true_response


def test_no_target():
    data_tmp = {
        'ca': 0,
        'sex': 1,
        'cp': 0,
        'trestbps': 120,
        'chol': 100,
        'fbs': 0,
        'restecg': 0,
        'age': 33,
        'thalach': 100,
        'exang': 0,
        'oldpeak': 1,
        'slope': 0,
        'thal': 0,
        'condition': 0
    }
    response = client.post('/predict', content=json.dumps(data_tmp))
    true_response = 'target should not be included'
    assert response.status_code == 400
    assert response.json()['detail'] == true_response


# для запуска тестов - python -m pytest tests
