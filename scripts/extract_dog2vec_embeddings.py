import os
import sys
import numpy as np
import torch
import soundfile as sf

# Allow Python to find Dog2Vec code
sys.path.append("dog2vec")

from extract_feature import FeatureExtractor

MODEL_PATH = "dog2vec/dog2vec_130K_9.pt"
TRAIN_DIR = "splits/train"
TEST_DIR = "splits/test"
OUTPUT_DIR = "embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

extractor = FeatureExtractor(
    model_path=MODEL_PATH,
    device=device,
    layer=9
)


def load_wav(path):
    audio, sample_rate = sf.read(path)

    # If stereo, convert to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    audio = torch.tensor(audio, dtype=torch.float32)

    return audio, sample_rate


def extract_dataset_embeddings(split_dir):
    embeddings = []
    labels = []
    file_paths = []

    for dog_name in sorted(os.listdir(split_dir)):
        dog_path = os.path.join(split_dir, dog_name)

        if not os.path.isdir(dog_path):
            continue

        for filename in sorted(os.listdir(dog_path)):
            if not filename.endswith(".wav"):
                continue

            wav_path = os.path.join(dog_path, filename)
            print("Extracting:", wav_path)

            audio, sample_rate = load_wav(wav_path)

            with torch.no_grad():
                features = extractor.extract(audio)

            # Dog2Vec returns frame-level features.
            # We average them into one embedding per bark clip.
            embedding = features.mean(dim=0).cpu().numpy()

            embeddings.append(embedding)
            labels.append(dog_name)
            file_paths.append(wav_path)

    return np.array(embeddings), np.array(labels), np.array(file_paths)


train_embeddings, train_labels, train_files = extract_dataset_embeddings(TRAIN_DIR)
test_embeddings, test_labels, test_files = extract_dataset_embeddings(TEST_DIR)

np.save(os.path.join(OUTPUT_DIR, "train_embeddings.npy"), train_embeddings)
np.save(os.path.join(OUTPUT_DIR, "train_labels.npy"), train_labels)
np.save(os.path.join(OUTPUT_DIR, "train_files.npy"), train_files)

np.save(os.path.join(OUTPUT_DIR, "test_embeddings.npy"), test_embeddings)
np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"), test_labels)
np.save(os.path.join(OUTPUT_DIR, "test_files.npy"), test_files)

print("Done.")
print("Train embeddings shape:", train_embeddings.shape)
print("Test embeddings shape:", test_embeddings.shape)