import os
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
        "train_pipeline",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=datetime(2022, 11, 27),
) as dag:
    split_data = DockerOperator(
        image="airflow-data-split",
        command="--input_dir /data/raw/{{ ds }} --output_train_dir /data/splitted/raw/train/{{ ds }} "
                "--output_val_dir /data/splitted/raw/val/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-split-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_PATH, target="/data", type='bind')]
    )

    preprocess_data = DockerOperator(
        image="airflow-preprocess",
        command="--input_dir /data/splitted/raw/train/{{ ds }} --output_dir /data/splitted/processed/train/{{ ds }} "
                "--val_dir /data/validation_artefacts/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-process-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_PATH, target="/data", type='bind')]
    )

    train = DockerOperator(
        image="airflow-train",
        command="--input_dir /data/splitted/processed/train/{{ ds }} "
                "--output_dir /data/validation_artefacts/{{ ds }}",
        network_mode="host",
        task_id="docker-airflow-train-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_PATH, target="/data", type='bind'),
                Mount(source=MLFLOW_RUNS_PATH, target="/mlflow_runs", type='bind')]
    )

    validation = DockerOperator(
        image="airflow-validation",
        command="--input_dir /data/splitted/raw/val/{{ ds }} "
                "--model_dir /data/validation_artefacts/{{ ds }}",
        network_mode="host",
        task_id="docker-airflow-validation",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_PATH, target="/data", type='bind'),
                Mount(source=MLFLOW_RUNS_PATH, target="/mlflow_runs", type='bind')]
    )

    split_data >> preprocess_data >> train >> validation
