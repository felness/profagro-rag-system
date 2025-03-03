FROM python:3.12.8-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/
COPY retriever/ /app/retriever/

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
