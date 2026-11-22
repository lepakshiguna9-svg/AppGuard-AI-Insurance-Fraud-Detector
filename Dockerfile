# Multi-stage production build for AppGuard AI
# Stage 1: Build Frontend Assets
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend & Serving Environment
FROM python:3.12-slim AS backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install minimal OS utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code, models, data, and deliverables
COPY appguard/ ./appguard/
COPY data/ ./data/
COPY artifacts/ ./artifacts/
COPY deliverables/ ./deliverables/

# Copy compiled frontend from frontend-builder
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Ensure appuser owns application directory
RUN chown -R appuser:appgroup /app

# Switch to non-root execution user
USER 10001

# Operational container health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/api/v1/health" || exit 1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn appguard.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
