# Base Image
FROM python:3.9-slim

# Working Directory
WORKDIR /app

# Install Python Packages
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy App Code
COPY . .

# Expose Dashboard Port
EXPOSE 8501

# Command - Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8501", "--workers", "2", "--timeout", "120", "app:app"]
