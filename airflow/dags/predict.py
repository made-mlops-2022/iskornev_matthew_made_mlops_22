from datetime import timedelta
from datetime import datetime
from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


CURRENT_MODEL_DATE = Variable.get("CURRENT_MODEL_DATE")
DATA_PATH = Variable.get("DATA_PATH")


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
        start_date=datetime(2022, 11, 28),
) as dag:

    predict = DockerOperator(
        image="airflow-predict",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/predictions/{{ ds }} "
                f"--model_dir /data/validation_artefacts/{CURRENT_MODEL_DATE}",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=DATA_PATH, target="/data", type='bind')]
    )