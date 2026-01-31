# Base Image (Reusing local producer image to avoid network pull)
FROM fintech-backend-project-kafka-producer-python:latest

# Working Directory
WORKDIR /app

# Install Python Packages
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy App Code
COPY . .

# Expose Streamlit Port
EXPOSE 8501

# Command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
