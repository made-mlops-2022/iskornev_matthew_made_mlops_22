FROM python:3.9-slim

COPY predict.py my_transformer.py requirements.txt /

RUN  python3 -m pip install --upgrade pip \
     && pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "predict.py"]