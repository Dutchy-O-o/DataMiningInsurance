# Dockerfile for Smart Insurance Advisor V2.0
# Uses Python 3.11 slim for smaller image size

FROM python:3.11-slim

# Install system dependencies needed by LightGBM, XGBoost, and SHAP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Flask default port
EXPOSE 5000

# Environment variables
ENV FLASK_APP=DataSet/app.py
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/stats')" || exit 1

# Run the Flask app
WORKDIR /app/DataSet
CMD ["python", "app.py"]
