# The llm-reserve API (Day 12): FastAPI + ONNX Runtime, no torch or xgboost.
# Needs the ONNX bundle in data/models/onnx/ (`make onnx`) before `docker compose up --build`.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt requirements-serve.in ./
RUN pip install -r requirements-serve.in -c requirements.txt

COPY src/ src/
COPY data/models/onnx/ data/models/onnx/

RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["uvicorn", "llm_reserve.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
