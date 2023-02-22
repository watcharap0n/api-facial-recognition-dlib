FROM python:3.9-slim

WORKDIR service_rec

COPY requirements.txt .

RUN apt-get update && apt-get -y install cmake protobuf-compiler libx11-dev
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY server .

WORKDIR ../

CMD ["uvicorn", "service_rec.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8080"]
