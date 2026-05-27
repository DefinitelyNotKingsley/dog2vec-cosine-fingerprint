import os
import sys
import numpy as np
import torch
import soundfile as sf

sys.path.append("dog2vec")
sys.path.append("../Fingerprinting/fairseq")

from extract_feature import FeatureExtractor

MODEL_PATH = "dog2vec/dog2vec_130K_9.pt"
PROFILES_PATH = "profiles/dog_profiles.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"

extractor = FeatureExtractor(
    model_path=MODEL_PATH,
    device=device,
    layer=9
)

dog_profiles = np.load(PROFILES_PATH, allow_pickle=True).item()


def load_wav(path):
    audio, sample_rate = sf.read(path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    return torch.tensor(audio, dtype=torch.float32)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if len(sys.argv) < 2:
    print("Usage:")
    print("python3 scripts/predict_similarity.py path/to/file.wav")
    sys.exit()

WAV_PATH = sys.argv[1]

audio = load_wav(WAV_PATH)

with torch.no_grad():
    features = extractor.extract(audio)

embedding = features.mean(dim=0).cpu().numpy()

scores = {}

for dog, profile in dog_profiles.items():
    score = cosine_similarity(embedding, profile)
    scores[dog] = score

predicted_dog = max(scores, key=scores.get)

print("\n===================================")
print("File:", WAV_PATH)
print("Predicted dog:", predicted_dog)
print("Cosine similarity:", round(scores[predicted_dog], 4))

print("\nAll similarity scores:")
for dog, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{dog}: {round(score, 4)}")

print("===================================")