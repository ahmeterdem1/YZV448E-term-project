FROM python:3.11-slim-buster

EXPOSE 8000

WORKDIR /app

COPY requirements.txt .
COPY data/ ./data/
# copy the data folder for training and testing

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
