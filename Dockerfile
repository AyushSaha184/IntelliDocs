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
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/dist /app/frontend/static

# Create necessary directories
RUN mkdir -p data/sessions data/documents data/chunks data/vector_store logs

# Download embedding model during build (saves ~2 min on cold start)
# sentence-transformers caches to ~/.cache/torch/sentence_transformers
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('BAAI/bge-m3'); \
print(f'Embedding model loaded: dim={model.get_sentence_embedding_dimension()}')"

# ── Environment variables for Render deployment ──
# These switch from LM Studio (local dev) to in-process models (container)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Embedding: use local sentence-transformers instead of LM Studio API
ENV EMBEDDING_PROVIDER=local
ENV EMBEDDING_MODEL=BAAI/bge-m3

# Reranker: use local CrossEncoder instead of LM Studio API
ENV RERANKER_PROVIDER=local
ENV RERANKER_MODEL=BAAI/bge-reranker-v2-m3

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