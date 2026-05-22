FROM python:3.11-slim

WORKDIR /app

# System dependencies:
#   gcc/g++       — compile some Python packages
#   libpq-dev     — psycopg2 (PostgreSQL client)
#   libfreetype6  — matplotlib / wordcloud fonts
#   fonts-dejavu  — wordcloud default font
#   curl          — Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    libpq-dev \
    libfreetype6-dev libpng-dev \
    fonts-dejavu \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create runtime directories
RUN mkdir -p data models

EXPOSE 8050

# Production: Gunicorn with 4 sync workers
# Dev override: docker compose run dashboard python app.py
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:server"]
