# Use official Python 3.10 slim base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies list
COPY requirements.txt .

# Install CPU-only PyTorch (uses --extra-index-url so PyPI dependencies like flit_core resolve properly)
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data during build phase to avoid runtime download latencies
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Copy application directories and source code
COPY src/ ./src/
COPY models/ ./models/
COPY reports/ ./reports/
COPY public/ ./public/
COPY data/ ./data/

# Expose service ports (Render default 10000, standard local 8000)
EXPOSE 10000 8000

# Start Uvicorn ASGI server (uses dynamic $PORT provided by Render, defaulting to 10000)
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-10000}"]
