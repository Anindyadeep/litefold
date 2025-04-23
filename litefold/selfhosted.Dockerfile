FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC \
    CUDA_DEVICE=cuda:0

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p /data/results /data/db
COPY . .
RUN ln -sf /data/results results && \
    touch /data/db/jobs.db


ENV SQLALCHEMY_DATABASE_URL=sqlite:////data/db/jobs.db
EXPOSE 8000
VOLUME ["/data/results", "/data/db"]
CMD ["python3", "selfhosted.py"]
