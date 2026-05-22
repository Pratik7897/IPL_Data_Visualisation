FROM python:3.11-slim

WORKDIR /app

# System deps for wordcloud / matplotlib
RUN apt-get update && apt-get install -y \
    gcc g++ libfreetype6-dev libpng-dev fonts-dejavu \
    curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data models

EXPOSE 8050
CMD ["python", "app.py"]
