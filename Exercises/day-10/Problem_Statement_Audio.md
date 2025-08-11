# Problem Statement: Audio Classification with Transformers on the ESC-50 Environmental Sound Dataset

### Objective

Build and evaluate a transformer-based audio classification model using the ESC-50 dataset. The task involves loading and preprocessing environmental audio clips, applying a pretrained transformer or fine-tuning it for classifying audio into 50 environmental sound classes, and analyzing model performance.

### Dataset

**ESC-50** (Environmental Sound Classification)

- Publicly available dataset of 2,000 labeled environmental audio recordings (5 seconds each) across 50 classes such as dog bark, rain, siren, and keyboard typing.
- Download URL: https://github.com/karolpiczak/ESC-50 (direct audio files in WAV format and metadata CSV)
- The dataset is well suited for benchmarking environmental audio classification models.


### Learning Objectives

- Load and preprocess short environmental audio clips (sample rate standardization, normalization).
- Extract audio features suitable for transformer models (raw waveform or learned embeddings).
- Use a pretrained audio transformer model (e.g., Wav2Vec 2.0, Hubert) as a feature extractor or fine-tune it for classification.
- Train, validate, and test the model on the ESC-50 dataset classes.
- Visualize training progress and assess model accuracy, confusion matrices, and class-wise performance.
- Investigate the attention patterns of the transformer to understand which audio segments influence classification.


### Tasks

1. **Dataset Preparation**
    - Download and unzip ESC-50.
    - Load metadata CSV to map audio file names to labels.
    - Implement audio loading pipeline (e.g., using librosa or torchaudio) to read and preprocess the clips uniformly.
2. **Feature Extraction and Modeling**
    - Use pretrained transformer audio models (e.g., `facebook/wav2vec2-base`, `superb/hubert-large-superb-ks`) from Hugging Face Transformers.
    - Convert audio clips into model inputs (waveform arrays or processed features).
    - Fine-tune on ESC-50 or extract fixed embeddings and train a classifier head.
3. **Training and Evaluation**
    - Split data into training and testing sets as per standard ESC-50 folds or create your own split.
    - Train the model and monitor learning curves.
    - Evaluate accuracy, per-class precision \& recall, and generate confusion matrix.
4. **Attention Visualization and Analysis (Optional)**
    - Visualize attention maps or gradients to interpret model focus on specific temporal audio segments.
    - Analyze attention differences across sound classes.
5. **Reporting**
    - Summarize data preprocessing challenges and solutions.
    - Present quantitative model results and qualitative insights from attention visualizations.
    - Discuss potential improvements like data augmentation or ensembling.

### Deliverables

- Python notebook or script implementing data loading, preprocessing, training, and evaluation.
- Visualizations of attention weights and model performance metrics.
- A brief report or markdown summarizing findings, challenges, and future directions.
