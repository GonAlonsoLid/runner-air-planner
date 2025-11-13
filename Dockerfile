FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
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

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: run Streamlit app
CMD ["streamlit", "run", "src/runner_air_planner/frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
