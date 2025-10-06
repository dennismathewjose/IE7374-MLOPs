# Base image
FROM python:3.10-slim

WORKDIR /app

COPY world-happiness-report-2021.csv .
COPY app/ app/
COPY app/requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "-m", "app.main"]
