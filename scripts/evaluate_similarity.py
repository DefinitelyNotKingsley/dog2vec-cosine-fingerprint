import os
import csv
import numpy as np

EMBEDDINGS_DIR = "embeddings"
PROFILES_PATH = "profiles/dog_profiles.npy"

X_test = np.load(os.path.join(EMBEDDINGS_DIR, "test_embeddings.npy"))
y_test = np.load(os.path.join(EMBEDDINGS_DIR, "test_labels.npy"))
test_files = np.load(os.path.join(EMBEDDINGS_DIR, "test_files.npy"))

dog_profiles = np.load(PROFILES_PATH, allow_pickle=True).item()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


correct = 0
total = len(X_test)

print("\nDetailed Predictions:\n")

for embedding, true_label, file_path in zip(X_test, y_test, test_files):

    scores = {}

    for dog, profile in dog_profiles.items():
        score = cosine_similarity(embedding, profile)
        scores[dog] = score

    predicted_dog = max(scores, key=scores.get)

    if predicted_dog == true_label:
        correct += 1

    print(f"{file_path}")
    print(f"True: {true_label}")
    print(f"Predicted: {predicted_dog}")
    print(f"Similarity: {scores[predicted_dog]:.4f}")

    print("All scores:")

    for dog, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dog}: {score:.4f}")

    print()


accuracy = correct / total

print("===================================")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Correct: {correct}/{total}")
print("===================================")

os.makedirs("results", exist_ok=True)

with open("results/similarity_results.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "file",
        "true_label",
        "predicted_label",
        "similarity_score"
    ])

    for embedding, true_label, file_path in zip(
        X_test,
        y_test,
        test_files
    ):

        scores = {}

        for dog, profile in dog_profiles.items():
            score = cosine_similarity(embedding, profile)
            scores[dog] = score

        predicted_dog = max(scores, key=scores.get)

        writer.writerow([
            file_path,
            true_label,
            predicted_dog,
            round(scores[predicted_dog], 4)
        ])

print("\nSaved CSV results to results/similarity_results.csv")