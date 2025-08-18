## Problem Statement: Versioning Large Satellite Image Datasets and Models

### Background

You are working on an environmental monitoring project to classify satellite images for land cover types (forest, water, urban, agriculture). Both the raw image dataset and trained deep learning models are large in size and change frequently as new satellite data arrives and models get retrained.

Managing and sharing these large datasets and model artifacts along with code versions in a reproducible and scalable manner is critical for your team’s collaboration and deployment pipeline.

***

### Dataset

Use the **EuroSAT dataset** — a labeled collection of Sentinel-2 satellite images covering 10 land use and land cover classes.

- Dataset available at:
https://github.com/phelber/eurosat
- Size: ~1.1 GB with 27,000 images in 13 spectral bands.

***

### Task

- Initialize a new Git repository and set up DVC to track datasets and model checkpoints.
- Download and version the EuroSAT dataset files using DVC.
- Write a training script to train a CNN model on the dataset and save checkpoints.
- Track model checkpoints and related artifacts (e.g., metrics files) with DVC.
- Use Git LFS alongside DVC to manage any other large files (such as raw images or pretrained weights).
- Demonstrate switching between different dataset or model versions using Git and DVC commands.
- (Optional) Configure a remote DVC storage (S3, GDrive) to share datasets and models across the team.

***

### Learning Outcomes

- Properly use DVC commands (`dvc add`, `dvc push`, `dvc checkout`) for managing large datasets.
- Collaborate effectively by versioning model artifacts and data with source code.
- Understand when and how to combine Git LFS and DVC for large file management.
- Gain skills critical for scalable, reproducible ML projects involving big data.

***

This exercise simulates a real-world ML workflow for teams handling large, evolving datasets and complex models requiring robust version control.
