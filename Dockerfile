FROM python:3.10

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install flask numpy opencv-python-headless

WORKDIR /app
COPY app.py .

EXPOSE 8080
CMD ["python", "app.py"]
