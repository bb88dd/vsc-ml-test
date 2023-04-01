# Specify base image (https://hub.docker.com/_/python)
FROM python:3.11-buster

# Copy requirements.txt and install.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy python file.
COPY rf_train_test.py ./

# Command.
CMD ["python", "./rf_train_test.py"]

# Steps to build and run in terminal:
# 1. docker build -t tag e.g. docker build -t rf-model
# 2. docker run tag e.g. docker run rf-model