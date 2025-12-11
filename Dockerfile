FROM python:3.10-slim

# system deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# upgrade pip / wheel før vi installerer - reduserer risiko for wheel-build errors
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
# Ensure numpy is installed first (requirements order matters)
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080
CMD ["python","app.py"]
