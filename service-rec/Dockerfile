FROM python:3.9-windowsservercore-ltsc2019

WORKDIR service_rec

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY server .

WORKDIR ../

CMD ["uvicorn", "service_rec.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8080"]
