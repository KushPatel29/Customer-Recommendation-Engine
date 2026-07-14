# Self-contained recommendation service image.
# Build runs the full pipeline (generate -> score -> evaluate) so the container
# serves real, holdout-evaluated recommendations with zero external state:
#
#   docker build -t rec-api .
#   docker run -p 8000:8000 rec-api
#   open http://localhost:8000/docs
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

COPY . .

# Bake the batch-scored artifacts into the image at build time.
RUN python data_generator/generate_sales_data.py \
 && python engine/recommend.py \
 && python analytics/customer_analytics.py \
 && python analytics/product_analytics.py

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
