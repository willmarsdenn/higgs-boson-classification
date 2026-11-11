# Target environment for Osprey (x64)
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas xgboost optuna scikit-learn pytest

# Copy the pipeline and tests
COPY src/ src/
COPY tests/ tests/
COPY data/ data/

RUN chmod -R 777 /app

# Force the pipeline to use Nvidia CUDA
ENV USE_GPU="true"

# Run the pipeline using the native OS Python
CMD ["python", "src/pipeline.py"]