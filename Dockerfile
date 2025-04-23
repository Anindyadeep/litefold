FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    curl \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first to ensure proper initialization
RUN pip install --no-cache-dir numpy==1.24.3 Cython==3.0.6

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies with specific options to handle conflicts
RUN pip install --no-cache-dir -r requirements.txt --use-pep517

# Create results directory
RUN mkdir -p results

# Copy application code
COPY litefold/ .

# Install uvicorn
RUN pip install uvicorn

# Expose the port the app runs on
EXPOSE 8178

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Run the application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8178"] 