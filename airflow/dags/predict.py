from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from params_and_paths import DATA_PATH, MLFLOW_RUNS_PATH, START_DATE, DEF_ARGS


with DAG(
        "predict",
        default_args=DEF_ARGS,
        schedule_interval="@daily",
        start_date=START_DATE,
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
