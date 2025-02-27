FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY controller.py .
RUN chmod +x controller.py

CMD ["python", "/app/controller.py"] 