FROM python:3.9-slim

COPY heart_cleveland_upload.csv requirements.txt get_data.py /

RUN  python3 -m pip install --upgrade pip \
     && pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "get_data.py"]