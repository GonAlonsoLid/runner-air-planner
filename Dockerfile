FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Configure Poetry: Don't create virtual env, install dependencies system-wide
RUN poetry config virtualenvs.create false

# Copy Poetry files
COPY pyproject.toml ./
COPY poetry.lock* ./

# Install dependencies with Poetry
# If poetry.lock doesn't exist, Poetry will generate it automatically
RUN poetry install --no-interaction --no-ansi --no-root || \
    (poetry lock --no-update && poetry install --no-interaction --no-ansi --no-root)

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create data directories (will be mounted as volumes)
RUN mkdir -p data/models data/raw data/interim data/processed

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command: run FastAPI app
CMD ["uvicorn", "runner_air_planner.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
