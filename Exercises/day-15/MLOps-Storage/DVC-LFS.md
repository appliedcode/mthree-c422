## Exercise: Data and Model Versioning with DVC \& Git LFS


***

### Objective

Practice managing large datasets and model artifacts in an image classification project using DVC and Git LFS, including downloading a public dataset.

***

### Scenario

You’ll classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)—a collection of 60,000 tiny images from 10 classes.

***

### Tasks

#### 1. **Project Setup**

- Create your project folder and initialize version control:

```bash
mkdir image-classification-dvc
cd image-classification-dvc
git init
python -m venv venv
source venv/bin/activate
```


#### 2. **Install Required Tools**

- Install DVC and Git LFS:

```bash
pip install dvc
git lfs install
pip install tensorflow keras  # For data loading, simulation
```


#### 3. **Download and Prepare Public Dataset**

- Download CIFAR-10 using Keras:

```python
# Save as download_dataset.py, then run: python download_dataset.py
import numpy as np
from tensorflow.keras.datasets import cifar10
import os

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
os.makedirs("data", exist_ok=True)
np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)
print("Dataset saved in ./data/")
```


***

#### 4. **Track Data with DVC**

- Initialize DVC and add the downloaded data:

```bash
dvc init
dvc add data/X_train.npy data/y_train.npy data/X_test.npy data/y_test.npy
git add data/*.dvc .gitignore
git commit -m "Add CIFAR-10 dataset tracked with DVC"
```


***

#### 5. **Simulate Model Training and Save Artifact**

- Write a script that “trains” (or simulates training) and saves a model:

```python
# Save as train.py, and run: python train.py
import numpy as np
from joblib import dump
# Dummy model: Save mean of X_train as 'model'
X_train = np.load("data/X_train.npy")
model = {"mean_pixel": np.mean(X_train)}
dump(model, "model/model.joblib")
print("Dummy model saved!")
```

- Track the model artifact:

```bash
mkdir model
dvc add model/model.joblib
git add model/model.joblib.dvc .gitignore
git commit -m "Add trained model tracked with DVC"
```


***

#### 6. **Version Large Files with Git LFS**

- Configure Git LFS:

```bash
git lfs track "*.npy"
git lfs track "*.joblib"
git add .gitattributes
git commit -m "Track npy and joblib files with Git LFS"
```


***

#### 7. **Experiment Tracking and Versioning**

- Each experiment (different data, preprocessing, or model) can be tracked with DVC:

```bash
# After retraining and saving new model artifact:
dvc add model/model_v2.joblib
git add model/model_v2.joblib.dvc
git commit -m "Add retrained model version"
```


***

#### 8. **(Optional) DVC Remote Storage**

- Add a remote storage (S3, GDrive, etc.) to sync large files:

```bash
dvc remote add -d myremote s3://my-dvc-bucket/
dvc push      # Upload data/artifacts to remote
```


***

#### 9. **Restore Data or Model Versions**

- To reproduce older results, checkout a previous commit and use DVC:

```bash
git checkout <commit-hash>
dvc checkout
```


***

### Deliverables

- A full Git repo with DVC and LFS configuration, dataset, and at least two tracked model versions.
- Brief notes on how data and models are versioned and how you would share/restore them.

***

## Questions

- What’s the difference between using DVC and Git LFS for dataset/model management?
- How does using a remote with DVC improve collaboration?

***

**This exercise enables reproducible ML development for large real-world datasets and models, with practical tools for data scientists and teams.**

