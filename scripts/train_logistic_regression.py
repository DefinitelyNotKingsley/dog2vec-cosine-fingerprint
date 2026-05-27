import os
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

EMBEDDINGS_DIR = "embeddings"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

X_train = np.load(os.path.join(EMBEDDINGS_DIR, "train_embeddings.npy"))
y_train = np.load(os.path.join(EMBEDDINGS_DIR, "train_labels.npy"))

X_test = np.load(os.path.join(EMBEDDINGS_DIR, "test_embeddings.npy"))
y_test = np.load(os.path.join(EMBEDDINGS_DIR, "test_labels.npy"))

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))

print("\nClass order:")
print(model.classes_)

joblib.dump(model, os.path.join(MODEL_DIR, "logistic_regression_dog2vec.pkl"))

print("\nModel saved to models/logistic_regression_dog2vec.pkl")