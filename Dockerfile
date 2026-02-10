# Dockerfile с поддержкой AMD64 (CUDA) и ARM64 (CPU)
ARG TARGETARCH

# === Выбор базового образа ===
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base-amd64
FROM python:3.10-slim AS base-arm64

# === Финальная стадия ===
FROM base-${TARGETARCH} AS final

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# === Установка Python и базовых пакетов ===
ARG TARGETARCH
RUN if [ "${TARGETARCH}" = "amd64" ]; then \
        echo "Setting up Python for AMD64 (CUDA)..."; \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            python3.10 \
            python3-pip \
            curl && \
        ln -sf /usr/bin/python3.10 /usr/bin/python && \
        ln -sf /usr/bin/pip3 /usr/bin/pip && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        echo "Python already available in ARM64 base image"; \
        apt-get update && \
        apt-get install -y --no-install-recommends curl && \
        rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

COPY requirements.docker.txt .

# === Установка зависимостей с учетом архитектуры ===
ARG TARGETARCH
RUN if [ "${TARGETARCH}" = "arm64" ]; then \
        echo "Installing CPU-only PyTorch for ARM64 (Mac)..."; \
        pip install --no-cache-dir \
            sentence-transformers \
            transformers \
            qdrant-client \
            fastapi \
            uvicorn[standard] \
            streamlit \
            requests && \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    else \
        echo "Installing CUDA PyTorch for AMD64 (NVIDIA GPU)..."; \
        pip install --no-cache-dir \
            --extra-index-url https://download.pytorch.org/whl/cu121 \
            -r requirements.docker.txt; \
    fi

COPY src/app/ ./
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 8501
ENTRYPOINT ["/entrypoint.sh"]