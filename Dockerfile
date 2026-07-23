FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/markdown data/chunks \
             data/vector_db/documents data/vector_db/vault \
             data/bm25

# Non-root user for security
RUN useradd -m -u 1000 rag && chown -R rag:rag /app
USER rag

# Pre-download reranker model so container startup is fast
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('Qwen/Qwen3-Reranker-0.6B', max_length=512)" || true

EXPOSE 8765

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
