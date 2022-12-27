import os
import logging
import pandas as pd


os.environ.setdefault('PATH_TO_MODEL', './data/model/model.pkl')
os.environ.setdefault('PATH_TO_TRANSFORMER', './data/transformer/trans.pkl')

logger = logging.getLogger("logging_service")


def get_model_response(input, model, transformer):
    data = pd.json_normalize(input.__dict__)
    data_processed = transformer.transform(data)
    prediction = model.predict(data_processed)
    if prediction is not None:
        logger.info('Model return prediction')
    proba = model.predict_proba(data_processed).max()
    proba = round(proba, 2)
    if prediction == 1:
        label = "Disease"
    else:
        label = "No disease"
    resp_dict = {
        'label': label,
        'prediction': int(prediction),
        'probability': proba
    }
    return resp_dict
