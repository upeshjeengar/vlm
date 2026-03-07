# Pull VLM from dockerhub: docker pull upeshj/vlm:1.0.0 (https://hub.docker.com/repository/docker/upeshj/vlm)
FROM python:3.12-slim

# Install system build tools and git (required for some packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the dependency definition files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install uv (the fast Python package manager)
RUN pip install --no-cache-dir uv==0.10.8

# Install exact dependencies as specified in uv.lock (no development deps)
RUN uv sync --no-dev --frozen

# Copy the rest of the project source code
COPY . .

# Optionally set a default command – adjust as needed for your application
# Here we assume an illustrative entry point script; replace with your actual entry script if different
CMD ["python", "inference.py"]
