from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from params_and_paths import DATA_PATH, MLFLOW_RUNS_PATH, START_DATE,\
    DEF_ARGS, VAL_ARTEFACTS, TRAIN_SIZE, MODEL_PARAMS


with DAG(
        "train_pipeline",
        default_args=DEF_ARGS,
        schedule_interval="@weekly",
        start_date=START_DATE,
) as dag:
    split_data = DockerOperator(
        image="airflow-data-split",
        command="--input_dir /data/raw/{{ ds }} --output_train_dir /data/splitted/raw/train/{{ ds }} "
                "--output_val_dir /data/splitted/raw/val/{{ ds }} "
                f"--train_size {TRAIN_SIZE}",
        network_mode="bridge",
        task_id="docker-airflow-split-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_PATH, target="/data", type='bind')]
    )

    preprocess_data = DockerOperator(
        image="airflow-preprocess",
        command="--input_dir /data/splitted/raw/train/{{ ds }} --output_dir /data/splitted/processed/train/{{ ds }} "
                f"--val_dir {VAL_ARTEFACTS}",
        network_mode="bridge",
        task_id="docker-airflow-process-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_PATH, target="/data", type='bind')]
    )

    train = DockerOperator(
        image="airflow-train",
        command="--input_dir /data/splitted/processed/train/{{ ds }} "
                f"--output_dir {VAL_ARTEFACTS} "
                f"-d {MODEL_PARAMS[0][0]} {MODEL_PARAMS[0][1]} "
                f"-d {MODEL_PARAMS[1][0]} {MODEL_PARAMS[1][1]} "
                f"-d {MODEL_PARAMS[2][0]} {MODEL_PARAMS[2][1]}",
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
                f"--model_dir {VAL_ARTEFACTS}",
        network_mode="host",
        task_id="docker-airflow-validation",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_PATH, target="/data", type='bind'),
                Mount(source=MLFLOW_RUNS_PATH, target="/mlflow_runs", type='bind')]
    )

    split_data >> preprocess_data >> train >> validation
