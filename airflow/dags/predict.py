from datetime import timedelta
from datetime import datetime
from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


DATA_PATH = Variable.get("DATA_PATH")
MLFLOW_RUNS_PATH = Variable.get("MLFLOW_RUNS_PATH")


default_args = {
    "owner": "matthewiskornev",
    "email": ["matveiiskornev@mail.ru"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=datetime(2022, 11, 27),
) as dag:

    predict = DockerOperator(
        image="airflow-predict",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/predictions/{{ ds }}",
        network_mode="host",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_PATH, target="/data", type='bind'),
                Mount(source=MLFLOW_RUNS_PATH, target="/mlflow_runs", type='bind')]
    )
