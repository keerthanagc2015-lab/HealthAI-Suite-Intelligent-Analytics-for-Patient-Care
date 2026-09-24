
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY deployment/requirements-runtime.txt /app/requirements-runtime.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements-runtime.txt

COPY . /app

EXPOSE 8000 8501

CMD ["uvicorn", "01_fastapi_app:app", "--app-dir", "/app/src/18_api", "--host", "0.0.0.0", "--port", "8000"]