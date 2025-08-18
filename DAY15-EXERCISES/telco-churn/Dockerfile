FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train.py app.py WA_Fn-UseC_-Telco-Customer-Churn.csv ./

RUN python train.py

EXPOSE 5000

CMD ["python", "app.py"]
