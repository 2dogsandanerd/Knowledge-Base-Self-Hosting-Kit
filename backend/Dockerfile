FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for Docling and document processing
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libgomp1 \
    curl \
    libxcb1 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (separate layer so code changes don't rebuild dependencies)
COPY . .

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
