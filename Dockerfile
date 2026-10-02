FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 FASTEMBED_CACHE_PATH=/app/.models

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
# Bake the index into the image: fetch the manual, embed it locally, persist Chroma.
RUN python -m app.ingest

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
