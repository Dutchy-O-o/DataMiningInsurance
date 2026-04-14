# Smart Insurance Advisor V2.0 — production container
# Multi-stage not needed: runtime deps are small enough.
FROM python:3.11-slim

# System libs required by XGBoost and LightGBM wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application payload (venv/docs are excluded via .dockerignore)
COPY src/    ./src/
COPY webapp/ ./webapp/
COPY data/   ./data/
COPY models/ ./models/
COPY run.py  ./

# Runtime config
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    DOCKER=1

EXPOSE 5000

# Simple health probe hitting the stats endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/api/stats').status == 200 else 1)"

CMD ["python", "run.py"]
