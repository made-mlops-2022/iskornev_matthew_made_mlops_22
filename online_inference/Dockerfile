FROM python:3.9-slim

COPY requirements.txt setup.py ./service/
COPY ./microservice ./service/microservice
COPY ./data/model/model.pkl ./data/transformer/trans.pkl ./service/microservice/

WORKDIR service

RUN  python3 -m pip install --upgrade pip \
     && pip install --no-cache-dir -r requirements.txt .

ENV PATH_TO_MODEL='microservice/model.pkl'
ENV PATH_TO_TRANSFORMER='microservice/trans.pkl'

EXPOSE 8000

CMD ["uvicorn", "microservice.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]


# адрес сервиса http://127.0.0.1:8000 или http://localhost:8000

# собрать образ docker build -t matthewiskornev/online_inference:v2 .

# запустить контейнер docker run -p8000:8000 -it --name service matthewiskornev/online_inference:v2

# docker image rm -f matthewiskornev/online_inference:v2