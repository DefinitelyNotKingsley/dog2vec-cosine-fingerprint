import os
import csv
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

EMBEDDINGS_DIR = "embeddings"
MODEL_PATH = "models/logistic_regression_dog2vec.pkl"

X_test = np.load(os.path.join(EMBEDDINGS_DIR, "test_embeddings.npy"))
y_test = np.load(os.path.join(EMBEDDINGS_DIR, "test_labels.npy"))
test_files = np.load(os.path.join(EMBEDDINGS_DIR, "test_files.npy"))

model = joblib.load(MODEL_PATH)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nDetailed Predictions:")
for file, true_label, pred_label, probs in zip(test_files, y_test, predictions, probabilities):
    confidence = np.max(probs) * 100
    print(f"{file}")
    print(f"True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.2f}%")
    print()
    
os.makedirs("results", exist_ok=True)

with open("results/prediction_results.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "file",
        "true_label",
        "predicted_label",
        "confidence"
    ])

    for file, true_label, pred_label, probs in zip(
        test_files,
        y_test,
        predictions,
        probabilities
    ):

        confidence = np.max(probs) * 100

        writer.writerow([
            file,
            true_label,
            pred_label,
            round(confidence, 2)
        ])

print("\nSaved CSV results to results/prediction_results.csv")