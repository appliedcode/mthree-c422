# Lab Exercise: Building and Automating an Image Classification Model with Docker and GitHub Actions


***

## Objective

Create an image classification model pipeline and automate the training, evaluation, Docker image build, and artifact management using GitHub Actions with CI/CD.

***

## Use Case Scenario

Train a model to classify handwritten digits from the MNIST dataset, containerize the training environment into a Docker image, and automate the full workflow including image build and artifact upload.

***

## Prerequisites

- Experience with Python, Git, Docker, Git, and GitHub basics
- Familiarity with machine learning and image classification
- A GitHub account and Docker installed locally

***

## Step 1: Setup GitHub Repository and Files

### 1. Create GitHub Repo

- Create a new repository `mnist-classifier-docker-ci`
- Clone it locally:

```bash
git clone https://github.com/your-username/mnist-classifier-docker-ci.git
cd mnist-classifier-docker-ci
```


### 2. Add Python Scripts, Dockerfile and Requirements

Create these files in the repo folder.

***

### train_mnist.py

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import joblib
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train_ohe = to_categorical(y_train, 10)
y_test_ohe = to_categorical(y_test, 10)

# Build a simple neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train_ohe, epochs=5, validation_data=(x_test, y_test_ohe))

# Save model using TensorFlow SavedModel format
model.save('mnist_model')

# Save test data for evaluation elsewhere
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)
```


***

### evaluate_mnist.py

```python
import numpy as np
import tensorflow as tf

# Load test data
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# Load saved model
model = tf.keras.models.load_model('mnist_model')

# Evaluate
loss, acc = model.evaluate(x_test, tf.keras.utils.to_categorical(y_test), verbose=0)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Loss: {loss:.4f}")
```


***

### requirements.txt

```
tensorflow
numpy
joblib
```


***

### Dockerfile

```Dockerfile
# Use official Python image
FROM python:3.9-slim

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Default command to run training script
CMD ["python", "train_mnist.py"]
```


***

### 3. Commit and push initial code

```bash
git add .
git commit -m "Initial commit with training, evaluation scripts, Dockerfile, and requirements"
git push origin main
```


***

## Step 2: Create GitHub Actions Workflow

### 1. Create workflow directory and file

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/mnist-docker-ci.yml` with:

```yaml
name: MNIST Classifier Docker CI/CD

on:
  push:
    branches: [ main ]

jobs:
  build_and_test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repo
      uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Build Docker image
      run: docker build -t mnist-classifier:latest .

    - name: Run training container
      run: docker run --rm mnist-classifier:latest

    - name: Run evaluation container
      run: docker run --rm mnist-classifier:latest python evaluate_mnist.py

    - name: Upload model artifact
      uses: actions/upload-artifact@v3
      with:
        name: mnist_model
        path: mnist_model
```


***

### 2. Commit and push workflow

```bash
git add .github/workflows/mnist-docker-ci.yml
git commit -m "Add GitHub Actions workflow to build Docker image, train and evaluate"
git push origin main
```


***

## Step 3: Verify Workflow Execution

- Navigate to your GitHub repository.
- Click the **Actions** tab.
- Monitor the workflow triggered by your push.
- Verify all steps complete successfully.
- Check that your model artifact `mnist_model` is uploaded.

***

## Optional Extensions

- Publish Docker image to Docker Hub or GitHub Container Registry.
- Add unit tests for scripts and integrate test jobs.
- Extend pipeline to push model to cloud storage or a model registry.
- Create REST API Docker container to serve predictions.

***

## Deliverables Checklist

- GitHub repo with:
    - `train_mnist.py`
    - `evaluate_mnist.py`
    - `Dockerfile`
    - `requirements.txt`
    - `.github/workflows/mnist-docker-ci.yml`
- Evidence of successful workflow runs and model artifact uploads.

***