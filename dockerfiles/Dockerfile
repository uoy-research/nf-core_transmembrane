# Use the official Python 3.8 slim image as a base (it already includes Python 3.8)
FROM --platform=linux/amd64 python:3.8

# Set the maintainer label
LABEL maintainer="sandyjmacdonald@gmail.com"

# Install system-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libhdf5-dev \
      build-essential \
      procps \
      && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install pytorchcrf dependencies from its requirements.txt
RUN pip install -r pytorchcrf/requirements.txt

# Now install the local pytorchcrf package
RUN pip install ./pytorchcrf

# Install the main package
RUN pip install .
