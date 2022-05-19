# FROM python:3.9-slim AS base
FROM pytorch/pytorch:latest AS base
RUN apt-get update && apt-get install -y --no-install-recommends gcc

FROM base as runtime

# # Copy virtual env
# COPY --from=base /.venv /.venv
# ENV PATH="/.venv/bin:$PATH"

# create ans switch to a new user
RUN useradd --create-home capstone
WORKDIR /home/capstone
USER capstone

# Install application into container
COPY . .

RUN python3 -m pip install -r requirements.txt

FROM runtime as app


CMD ["pip", "list"]