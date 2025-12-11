FROM python:3.10-slim

# System deps: OpenCV libs, poppler (PDF -> image), tesseract (OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libtiff5-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python packaging safety
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first to avoid wheel rebuild issues
RUN pip install --no-cache-dir numpy==1.24.4

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Expose port
EXPOSE 8080

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=3s CMD curl --fail http://localhost:8080/health || exit 1

CMD ["python", "app.py"]
