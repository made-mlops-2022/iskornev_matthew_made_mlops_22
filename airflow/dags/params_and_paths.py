from airflow.models import Variable
from datetime import timedelta
from datetime import datetime
from airflow.utils.email import send_email_smtp


DATA_PATH = Variable.get("DATA_PATH")
MLFLOW_RUNS_PATH = Variable.get("MLFLOW_RUNS_PATH")
START_DATE = datetime(2022, 11, 27)
VAL_ARTEFACTS = "/data/validation_artefacts/{{ ds }}"
TRAIN_SIZE = 0.75

MODEL_PARAMS = [
    ['n_estimators', 150],
    ["max_depth", 5],
    ['random_state', 42]
]


def custom_failure_function(context):
    """Define custom failure notification behavior"""
    dag_run = context.get('dag_run')
    msg = "DAG ran unsuccessfully"
    subject = f"DAG {dag_run} has failed"
    send_email_smtp(to=DEF_ARGS['email'], subject=subject, html_content=msg)


DEF_ARGS = {
    "owner": "matthewiskornev",
    "email": ["dirijablvspovar@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    'on_failure_callback': custom_failure_function
}
