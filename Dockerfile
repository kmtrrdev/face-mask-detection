# ────────────────────────────────────────────────────────────────
# Face Mask Detection — Dockerfile
# Runs FastAPI (port 8000) + Streamlit (port 8501)
# ────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY streamlit_app.py .
COPY mask_detector.pth .
COPY start.sh .

RUN chmod +x start.sh

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["./start.sh"]
