# Multi-stage build for RAG Assistant on Render.com
# Stage 1: Build Frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy frontend source
COPY frontend/ ./

# Build frontend
RUN npm run build

# Stage 2: Python Backend
FROM python:3.11-slim

# Install system dependencies (libmagic for unstructured, poppler for PDFs)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy slim requirements and install (no PyTorch / local models)
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# Copy application code
COPY . .

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/dist /app/frontend/static

# Create necessary directories
RUN mkdir -p data/sessions data/documents data/chunks data/vector_store logs

# ── Environment variables for Render deployment ──
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Embedding: use NVIDIA API (optimized for scale)
ENV EMBEDDING_PROVIDER=nvidia
ENV EMBEDDING_MODEL=baai/bge-m3
ENV EMBEDDING_DIMENSION=1024

# LLM: keep OpenRouter (or set GEMINI_API_KEY via Render dashboard)
# Set these via Render Environment Variables dashboard:
#   OPENROUTER_API_KEY, GEMINI_API_KEY, DATABASE_URL, etc.

# Expose port (Render injects PORT env variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Start command — shell form so ${PORT} is expanded at runtime
CMD sh -c "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"