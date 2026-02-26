# ElectroAnatomyCloud — Dockerfile
#
# Build:   docker build -t electroanatomycloud .
# Run:     docker run -p 8050:8050 electroanatomycloud
# Compose: docker-compose up --build
#
# The app is served at http://localhost:8050
# Data directories are mounted as read-only volumes via docker-compose.yml.

FROM python:3.10-slim

# libgomp1 is required by Open3D on Linux.
# libgl1-mesa-glx was renamed to libgl1 in Debian trixie (python:3.10-slim base).
# libglib2.0-0 and libglib2.0-dev are needed by some scikit-image operations.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (before copying code) so Docker layer cache
# is reused when only source files change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full package
COPY . .

# The app is launched from the ElectroAnatomyCloud/ directory so that
# relative imports within EAM/ (e.g. `from graph import *`) resolve correctly.
WORKDIR /app

EXPOSE 8050

# Use the app/ subdirectory as the working directory at runtime so that
# `import pipeline` and `import callbacks` resolve without a package prefix.
CMD ["python", "app/app.py"]
