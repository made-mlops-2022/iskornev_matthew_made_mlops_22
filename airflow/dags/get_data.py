import os
from datetime import timedelta
from datetime import datetime
from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


DATA_PATH = Variable.get("DATA_PATH")


default_args = {
    "owner": "matthewiskornev",
    "email": ["matveiiskornev@mail.ru"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "get_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=datetime(2022, 11, 28),
) as dag:
    get_data = DockerOperator(
        image="airflow-get-data",
        command="--output_dir /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-get-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source=DATA_PATH, target="/data", type='bind')]
    )