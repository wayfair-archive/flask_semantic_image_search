FROM debian:stretch-slim

MAINTAINER Cung Tran "minishcung@gmail.com"

RUN apt-get update -y && apt-get install -y \
        gnupg \
        curl

RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" \
        | tee /etc/apt/sources.list.d/tensorflow-serving.list \
        && \
        curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg \
        | apt-key add - \
        && \
        apt-get update -y \
        && \
        apt-get install -y tensorflow-model-server \
        && \
        apt-get clean

COPY . /app
WORKDIR /app
