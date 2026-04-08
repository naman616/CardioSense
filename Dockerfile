FROM python:3.10-slim

WORKDIR /code

# libgomp1 is required by PyTorch (OpenMP threading)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/       ./api/
COPY app/       ./app/
COPY src/       ./src/
COPY frontend/  ./frontend/
COPY data/      ./data/
COPY results/   ./results/

# Hugging Face Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
