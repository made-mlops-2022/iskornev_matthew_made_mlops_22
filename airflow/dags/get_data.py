from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from params_and_paths import DATA_PATH, DEF_ARGS, START_DATE


with DAG(
        "get_data",
        default_args=DEF_ARGS,
        schedule_interval="@daily",
        start_date=START_DATE,
) as dag:
    get_data = DockerOperator(
        image="airflow-get-data",
        command="--output_dir /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-get-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_PATH, target="/data", type='bind')]
    )
