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

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 3000

# Command to activate the virtual environment, install uvicorn, and run the app
CMD ["/bin/bash", "-c", "source .venv/bin/activate && pip install uvicorn && uvicorn src.server.app:app --host 0.0.0.0 --port 3000"]
