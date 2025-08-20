# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# expose port
EXPOSE 8000
# model path is defaulted to models/heart_pipeline.pkl
ENV MODEL_PATH=models/heart_pipeline.pkl
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
