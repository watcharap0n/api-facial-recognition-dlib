FROM python:3.9-slim

WORKDIR service_rec

COPY requirements.txt .

RUN apt-get update && apt-get -y install cmake protobuf-compiler libsndfile

RUN pip install --upgrade pip
RUN pip install -r requirements.txt \
    && rm -rf /root/.cache/pip \

COPY server .

WORKDIR ../

CMD ["uvicorn", "service_rec.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8080"]
