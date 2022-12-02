import pytest
from airflow.models import DagBag


@pytest.fixture()
def dagbag():
    return DagBag()


def assert_dag_dict_equal(source, dag):
    assert dag.task_dict.keys() == source.keys()
    for task_id, downstream_list in source.items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)


def test_dag_get_data_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="get_data")
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 1


def test_dag_train_pipeline_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="train_pipeline")
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 4


def test_dag_predict_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="predict")
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 1


def test_dag_get_data_dict(dagbag):
    dag = dagbag.get_dag(dag_id="get_data")
    assert_dag_dict_equal(
        {
            "docker-airflow-get-data": [],
        },
        dag,
    )


def test_dag_train_pipeline_dict(dagbag):
    dag = dagbag.get_dag(dag_id="train_pipeline")
    assert_dag_dict_equal(
        {
            "docker-airflow-split-data": ["docker-airflow-process-data"],
            "docker-airflow-process-data": ["docker-airflow-train-data"],
            "docker-airflow-train-data": ["docker-airflow-validation"],
            "docker-airflow-validation": []
        },
        dag,
    )


def test_dag_predict_dict(dagbag):
    dag = dagbag.get_dag(dag_id="predict")
    assert_dag_dict_equal(
        {
            "docker-airflow-predict": [],
        },
        dag,
    )
