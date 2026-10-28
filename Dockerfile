FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY main.py .
COPY data/raw ./data/raw

RUN pip install --no-cache-dir -e .

CMD ["python", "main.py"]