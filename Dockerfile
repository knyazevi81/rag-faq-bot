FROM python:3.10-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt update -y && \
    apt install -y python3-dev gcc musl-dev
RUN apt-get update && apt-get install -y netcat

ADD requirements.txt /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app