
# ===================================================
# File: Dockerfile (optional)
# ===================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment
RUN conda env create -f /tmp/environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "mtl-gnn-dta", "/bin/bash", "-c"]

# Copy project files
WORKDIR /app
COPY . .

# Install package
RUN pip install -e .

# Set the default command
CMD ["conda", "run", "-n", "mtl-gnn-dta", "python", "examples/quick_start.py"]