FROM python:3.9-slim

COPY validation.py my_transformer.py requirements.txt /

RUN  python3 -m pip install --upgrade pip \
     && pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "validation.py"]