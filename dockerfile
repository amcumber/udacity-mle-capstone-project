FROM pytorch/pytorch:latest as base

RUN conda install --quiet --yes jupyter 
FROM base as runtime

# create ans switch to a new user
RUN useradd --create-home capstone
WORKDIR /home/capstone
USER capstone

# Install application into container
COPY . .

RUN python3 -m pip install -r requirements.txt

FROM runtime as app