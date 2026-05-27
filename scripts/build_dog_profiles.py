import os
import numpy as np

EMBEDDINGS_DIR = "embeddings"
OUTPUT_DIR = "profiles"

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_train = np.load(os.path.join(EMBEDDINGS_DIR, "train_embeddings.npy"))
y_train = np.load(os.path.join(EMBEDDINGS_DIR, "train_labels.npy"))

dog_profiles = {}

for dog in sorted(set(y_train)):
    dog_embeddings = X_train[y_train == dog]
    profile = dog_embeddings.mean(axis=0)
    dog_profiles[dog] = profile

    print(f"{dog}: profile created from {len(dog_embeddings)} bark clips")

np.save(os.path.join(OUTPUT_DIR, "dog_profiles.npy"), dog_profiles)

print("\nSaved dog profiles to profiles/dog_profiles.npy")