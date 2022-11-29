#1/bin/bash
export DATA_PATH=$(pwd)/data
export MLFLOW_RUNS_PATH=$(pwd)/mlflow_runs
export CURRENT_MODEL_DATE="2022-11-28"

export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")

docker-compose up --build