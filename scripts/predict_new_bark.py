import os
import sys
import numpy as np
import torch
import soundfile as sf
import joblib

sys.path.append("dog2vec")

from extract_feature import FeatureExtractor

MODEL_PATH = "dog2vec/dog2vec_130K_9.pt"
CLASSIFIER_PATH = "models/logistic_regression_dog2vec.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

extractor = FeatureExtractor(
    model_path=MODEL_PATH,
    device=device,
    layer=9
)

classifier = joblib.load(CLASSIFIER_PATH)


def load_wav(path):
    audio, sample_rate = sf.read(path)

    # stereo → mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    audio = torch.tensor(audio, dtype=torch.float32)

    return audio


# -------- command line argument --------

if len(sys.argv) < 2:
    print("Usage:")
    print("python3 scripts/predict_new_bark.py path/to/file.wav")
    sys.exit()

WAV_PATH = sys.argv[1]

# --------------------------------------

audio = load_wav(WAV_PATH)

with torch.no_grad():
    features = extractor.extract(audio)

embedding = features.mean(dim=0).cpu().numpy().reshape(1, -1)

prediction = classifier.predict(embedding)[0]
probabilities = classifier.predict_proba(embedding)[0]

best_index = np.argmax(probabilities)
confidence = probabilities[best_index]

print("\n===================================")
print("File:", WAV_PATH)
print("Predicted dog:", prediction)
print("Confidence:", round(confidence * 100, 2), "%")

print("\nAll confidence scores:")
for dog, prob in zip(classifier.classes_, probabilities):
    print(f"{dog}: {round(prob * 100, 2)}%")

print("===================================")