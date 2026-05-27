# Dog Bark Fingerprinting using Dog2Vec

## Overview

This project explores dog bark fingerprinting using Dog2Vec embeddings with both supervised classification and cosine similarity profile matching.

The goal of the project is to identify which dog produced a bark audio clip by learning unique vocal patterns from barking sounds.

The system uses:
- Dog2Vec for bark feature extraction
- Logistic Regression as a baseline classifier
- Cosine Similarity Profile Matching as a fingerprinting system

---

## Motivation

Traditional audio models are usually trained on human speech or general audio events. However, dog vocalizations contain species-specific acoustic patterns that may not be captured well by generic models.

Dog2Vec is a pretrained canine-specific bioacoustic embedding model trained on thousands of hours of dog barking audio. This project investigates whether Dog2Vec embeddings can act as a “voice fingerprint” for identifying individual dogs.

---

## Pipeline

### Baseline Classification System

```text
Bark WAV
↓
Dog2Vec
↓
768-D Embedding
↓
Logistic Regression
↓
Predicted Dog ID
```

### Fingerprinting / Similarity Matching System

```text
Bark WAV
↓
Dog2Vec
↓
768-D Embedding
↓
Cosine Similarity
↓
Closest Dog Profile
```

---

## Project Structure

```text
Fingerprinting_similarity/
├── data_barks/
├── dog2vec/
├── embeddings/
├── fairseq/
├── models/
├── profiles/
├── results/
├── scripts/
├── splits/
└── README.md
```

---

## Dataset

- Bark audio clips stored as `.wav`
- Organized by dog identity
- Dataset split into:
  - 80% training
  - 20% testing

Current dataset:
- 5 dogs
- 38 bark clips total

---

## Models Used

### 1. Dog2Vec

Type:
- Self-supervised audio embedding model

Purpose:
- Extract dog-specific bark representations

Output:
- 768-dimensional embedding vector

Dog2Vec captures:
- pitch
- bark texture
- rhythm
- acoustic identity patterns
- vocal characteristics

Reason for choosing Dog2Vec:
- Specifically trained on dog vocalizations
- Better suited for canine bioacoustics than generic speech models
- Produces strong bark fingerprint embeddings

---

### 2. Logistic Regression

Type:
- Supervised classification model

Purpose:
- Predict dog identity from embeddings

Advantages:
- Lightweight baseline model
- Fast training
- Works well on small datasets
- Produces confidence probabilities

---

### 3. Cosine Similarity Profile Matching

Type:
- Embedding similarity / fingerprint matching system

Purpose:
- Compare a new bark embedding against stored dog voice profiles

Method:
- Average embeddings from each dog into one profile
- Compare new bark embeddings using cosine similarity
- Highest similarity score determines prediction

Reason for using cosine similarity:
- More similar to a real fingerprint database system
- Easier to scale to larger datasets
- New dogs can be added without fully retraining a classifier
- Directly compares vocal similarity between dogs

---

## Results

### Logistic Regression

Accuracy:
```text
88.89%
```

### Cosine Similarity Profile Matching

Accuracy:
```text
88.89%
```

Both systems correctly identified:
```text
8 / 9 test bark clips
```

Example similarity output:

```text
dog_5: 0.6317
dog_2: 0.4931
dog_3: 0.4859
dog_4: 0.4466
dog_1: 0.1390
```

Prediction:
```text
dog_5
```

---

## Scripts

### Split Dataset

```bash
python3 scripts/split_data.py
```

### Extract Dog2Vec Embeddings

```bash
python3 scripts/extract_dog2vec_embeddings.py
```

### Train Logistic Regression

```bash
python3 scripts/train_logistic_regression.py
```

### Predict Using Logistic Regression

```bash
python3 scripts/predict_new_bark.py path/to/file.wav
```

### Build Dog Profiles

```bash
python3 scripts/build_dog_profiles.py
```

### Predict Using Cosine Similarity

```bash
python3 scripts/predict_similarity.py path/to/file.wav
```

### Evaluate Cosine Similarity System

```bash
python3 scripts/evaluate_similarity.py
```

---

## Future Work

Potential future improvements include:
- Larger dataset with more dogs
- Real-world noisy audio testing
- Unknown dog detection thresholding
- PCA / t-SNE embedding visualization
- Multi-bark profile aggregation
- Compare Dog2Vec vs HuBERT vs wav2vec2
- Deep learning classification models
- Multi-dog audio localization integration

---

## Research Goal

The long-term goal of this project is to develop a scalable dog vocal fingerprinting system capable of identifying individual dogs from barking audio in real-world environments.