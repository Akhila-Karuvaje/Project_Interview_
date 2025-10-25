FROM python:3.10.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
RUN python -c "import whisper; whisper.load_model('tiny', download_root='/tmp/whisper_cache')"

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads /tmp/transformers_cache /tmp/torch_cache /tmp/huggingface_cache /tmp/whisper_cache

# Set environment variables
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV TORCH_HOME=/tmp/torch_cache
ENV HF_HOME=/tmp/huggingface_cache

# Expose port
EXPOSE 10000

# Run with aggressive timeout and memory settings
CMD gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --timeout 600 \
    --workers 1 \
    --worker-class sync \
    --worker-tmp-dir /dev/shm \
    --preload \
    --log-level info
