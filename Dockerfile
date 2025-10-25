FROM python:3.10.13-slim

WORKDIR /app

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \FROM python:3.10.13-slim

WORKDIR /app

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker cache optimization
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Pre-download Whisper tiny model at build time
RUN python -c "import whisper; whisper.load_model('tiny', download_root='/tmp/whisper_cache')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads /tmp/transformers_cache /tmp/torch_cache /tmp/huggingface_cache

# Expose port
EXPOSE 10000

# Run with gunicorn - increased timeout for video processing
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --preload --worker-class sync
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 10000

# Run with gunicorn and increase timeout for safety
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app", "--timeout", "120"]
