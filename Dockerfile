FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install numpy requests flask

# Copy application files
COPY ./app /app

# Command will be overridden in docker-compose
CMD ["python", "-u", "worker.py"]