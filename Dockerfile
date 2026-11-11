# Target environment for Osprey (x64)
FROM --platform=linux/amd64 python:3.11-slim

# Install uv for package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency lockfiles
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Copy the pipeline and tests
COPY src/ src/
COPY tests/ tests/

# Force the pipeline to use Nvidia CUDA
ENV USE_GPU="true"

# Run the pipeline
CMD ["uv", "run", "python", "src/pipeline.py"]
