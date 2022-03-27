# syntax=docker/dockerfile:1

FROM python:3.8
ENV PYTHONUNBUFFERRED=1

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
COPY ./requirements.txt requirements.txt 
RUN pip install -r requirements.txt