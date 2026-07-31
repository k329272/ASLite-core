FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn[standard] pydantic

COPY . .
RUN if [ ! -f models/asl_recognizer.xml ] || [ ! -f models/asl_recognizer.bin ] || [ ! -f models/gloss_labels.json ] || [ ! -f models/G.pth ]; then bash ./download; fi

EXPOSE 7860
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-7860}"]
