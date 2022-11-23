import json
import logging
import os
import pandas as pd
import requests
from microservice.entities import FLOAT_FEATURES


os.environ.setdefault('PATH_TO_DATA', './data/synthetic_data.csv')
os.environ.setdefault('URL', 'http://127.0.0.1:8000/predict')
TARGET = 'condition'

logger = logging.getLogger("logging_requests")
logger.setLevel(logging.INFO)
formatter_stdout = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                     datefmt="%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter_stdout)
logger.addHandler(stream_handler)


def main():
    df = pd.read_csv(os.getenv('PATH_TO_DATA'))
    df.drop(columns=TARGET, inplace=True)
    for row in df.iterrows():

        row_int = row[1].astype(int)

        row_json = row_int.to_dict()
        for name in FLOAT_FEATURES:
            row_json[name] = row[1][name]

        response = requests.post(
            os.getenv('URL'),
            json.dumps(row_json)
        )
        logger.info(f"status code: {response.status_code}")
        logger.info(f"send data: {row_json}")
        logger.info(f"response: {response.json()}\n")


if __name__ == '__main__':
    main()
