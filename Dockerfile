FROM python:3.10-slim-bullseye

COPY ./DenoisingAE.py .
COPY test.py .
COPY requirements.txt .
COPY config.cfg .
RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python3", "test.py" ]