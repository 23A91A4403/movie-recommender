FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt


EXPOSE 8000

CMD ["sh", "-c", "python src/data_preprocessing.py && python src/user_based_cf.py && python src/svd_model.py && python src/content_based_model.py && python src/evaluate_models.py && python src/cold_start.py && uvicorn src.api:app --host 0.0.0.0 --port 8000"]
