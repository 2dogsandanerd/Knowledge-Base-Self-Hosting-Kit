FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for Docling and document processing
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Docker BuildKit caches pip packages automatically between builds
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application code (separate layer so code changes don't rebuild dependencies)
COPY . .

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
