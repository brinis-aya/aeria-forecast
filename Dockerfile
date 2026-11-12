FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY aeria_forecast/ aeria_forecast/
COPY models/ models/
COPY data/ data/

RUN useradd --create-home --uid 10001 aeria
USER aeria

EXPOSE 8083
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8083/health')"
# DW_URL (optional): read live measurements from the Aeria warehouse instead of the bundled dataset
CMD ["uvicorn", "aeria_forecast.api:app", "--host", "0.0.0.0", "--port", "8083", "--no-server-header"]
