FROM python:3.13-slim

# Run as uid 1000 (required by Hugging Face Spaces, good practice anywhere).
RUN useradd -m -u 1000 user
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 FASTEMBED_CACHE_PATH=/app/.models

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user app ./app
RUN chown user /app
USER user
# Bake the index into the image: fetch the manual, embed it locally, persist Chroma.
RUN python -m app.ingest

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
