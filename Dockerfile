# Multi-stage build for optimized image size
FROM python:3.11-slim as builder

WORKDIR /app

# Install git and other build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ensure user-installed scripts are on PATH during build
ENV PATH=/root/.local/bin:$PATH

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install git (needed for git operations at runtime)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY main.py .
COPY libs/ ./libs/
COPY .env* ./

# Make sure scripts in .local are in PATH
ENV PATH=/root/.local/bin:$PATH

# Create directory for knowledge base and vector DB
RUN mkdir -p /app/knowledge-base /app/vector_db

# Expose Gradio default port
EXPOSE 7860

# Health check endpoint (Gradio has built-in health at root)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860').read()"

# Run the application
CMD ["python", "main.py"]
