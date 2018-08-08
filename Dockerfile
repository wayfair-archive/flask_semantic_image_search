FROM python:3.6-slim-stretch

MAINTAINER Cung Tran "minishcung@gmail.com"

RUN apt-get update -y && apt-get install -y \
        build-essential \
        libblas-dev \
        liblapack-dev \
        && \
        apt-get clean

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
