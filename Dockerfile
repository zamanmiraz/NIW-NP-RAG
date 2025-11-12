# Use lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy dependencies first (for Docker caching)
COPY requirements_upd.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_upd.txt

# Copy your RAG application code
COPY . .

# Expose FastAPI port
EXPOSE 8080

# Default command to start FastAPI server
CMD ["uvicorn", "niw_np_rag.app.main:app", "--host", "0.0.0.0", "--port", "8080"]

