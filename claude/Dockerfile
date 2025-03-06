FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install numpy

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command will be specified in docker-compose.yml
CMD ["python", "-u", "/app/coordinator.py"]