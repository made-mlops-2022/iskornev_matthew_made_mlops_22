FROM python:3.9-slim

COPY preprocess_data.py requirements.txt my_transformer.py /

RUN  python3 -m pip install --upgrade pip \
     && pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "preprocess_data.py"]