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

# Copy Poetry files and source code (needed to install the package)
COPY pyproject.toml ./
COPY poetry.lock* ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install dependencies AND the package itself
# If poetry.lock doesn't exist, Poetry will generate it automatically
RUN poetry install --no-interaction --no-ansi || \
    (poetry lock --no-update && poetry install --no-interaction --no-ansi)

# Create data directories (will be mounted as volumes)
RUN mkdir -p data/models data/raw data/interim data/processed

# Expose API port (Render will use PORT env var)
EXPOSE 8000

# Health check (uses PORT env var if available)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD sh -c "curl -f http://localhost:${PORT:-8000}/api/health || exit 1"

# Default command: run FastAPI app
# Use PORT environment variable if available (for Render), otherwise default to 8000
CMD sh -c "uvicorn runner_air_planner.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"
