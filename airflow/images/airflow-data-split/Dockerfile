FROM python:3.9-slim

COPY requirements.txt split_data.py /

RUN  python3 -m pip install --upgrade pip \
     && pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "split_data.py"]