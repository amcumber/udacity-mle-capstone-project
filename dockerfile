# FROM python:3.9-slim AS base
FROM jupyter/scipy-notebook:latest as base

RUN conda install --quiet --yes pytorch torchvision -c soumith
FROM base as runtime

# create ans switch to a new user
RUN useradd --create-home capstone
WORKDIR /home/capstone
USER capstone

# Install application into container
COPY . .

RUN python3 -m pip install -r requirements.txt

FROM runtime as app