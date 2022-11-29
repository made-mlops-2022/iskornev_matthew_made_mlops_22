FROM python:3.7-slim

WORKDIR /mlflow/

RUN  python3 -m pip install --upgrade pip \
     && pip --no-cache-dir install mlflow

CMD ["mlflow", "server", \
# connect backend database
 "--backend-store-uri", "sqlite:///:memory", \
# choose local dir as an artifacts store
 "--default-artifact-root", "./mlflow_runs", \
# specify host and port
 "--host", "0.0.0.0", "--port", "5050"]