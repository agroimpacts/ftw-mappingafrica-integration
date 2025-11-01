FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# System deps (adjust if you need extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl gcc g++ libglib2.0-0 gdal-bin libgdal-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

# Copy packaging files first for better cache usage (if present)
COPY pyproject.toml setup.cfg setup.py /opt/app/ 2>/dev/null || true

RUN pip install --upgrade pip setuptools wheel

# Install project (editable if project files are copied/mounted later)
# This will succeed if setup.py/pyproject exists; if not, it's harmless.
RUN pip install -e . || true

# Copy repo
COPY . /opt/app
WORKDIR /opt/app

# Ensure required host-mounted dirs exist
RUN mkdir -p /data /training_logs

# Tiny entrypoint
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]