# Use a lightweight Python 3.10 base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt requirements.txt

# Create and install dependencies in the virtual environment
RUN python -m venv .venv && \
    ./.venv/bin/pip install --no-cache-dir --upgrade pip && \
    ./.venv/bin/pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding and rerank models into the image. src/server/utils.py
# loads both at import, so without this every API replica and every worker fetches
# ~500MB from HuggingFace on boot -- 10-30s before it can serve or consume anything,
# and a failed fetch means the container never starts. Its own layer so it is only
# rebuilt when requirements change.
ENV HF_HOME=/app/.cache/huggingface
RUN ./.venv/bin/python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 3000

# uvicorn comes from requirements.txt; installing at boot made every start depend on
# PyPI being reachable.
CMD ["/app/.venv/bin/uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "3000"]
