# Dog Bark Fingerprinting using Dog2Vec

## Overview

This project builds a dog bark fingerprinting system using:

- Dog2Vec embeddings
- Logistic Regression classification
- Bark audio clips in `.wav` format

The goal is to identify which dog produced a bark sound.

---

## Pipeline

```text
Dog Bark WAV
↓
Dog2Vec Feature Extraction
↓
768-Dimensional Embedding
↓
Logistic Regression Classifier
↓
Dog Identity Prediction
```

---

## Project Structure

```text
Fingerprinting/
├── data_barks/
├── dog2vec/
├── embeddings/
├── models/
├── results/
├── scripts/
├── splits/
└── README.md

---

## Results

- Test Accuracy: 88.89%
- Dog2Vec embedding size: 768
- Classifier: Logistic Regression

Example prediction:

```text
Predicted dog: dog_5
Confidence: 64.71%
```