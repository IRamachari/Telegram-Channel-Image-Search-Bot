FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y gcc g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY download_model.py .
RUN python download_model.py

COPY . .
RUN mkdir -p /app/chroma_db /app/temp_images && \
    chmod +x /app/entrypoint.sh

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
