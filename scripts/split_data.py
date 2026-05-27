import os
import shutil
from sklearn.model_selection import train_test_split

DATA_DIR = "data_barks"
OUTPUT_DIR = "splits"

train_dir = os.path.join(OUTPUT_DIR, "train")
test_dir = os.path.join(OUTPUT_DIR, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for dog_name in os.listdir(DATA_DIR):
    dog_path = os.path.join(DATA_DIR, dog_name)

    if not os.path.isdir(dog_path):
        continue

    wavs = [
        f for f in os.listdir(dog_path)
        if f.endswith(".wav")
    ]

    train_files, test_files = train_test_split(
        wavs,
        test_size=0.2,
        random_state=42
    )

    os.makedirs(os.path.join(train_dir, dog_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, dog_name), exist_ok=True)

    for file in train_files:
        shutil.copy(
            os.path.join(dog_path, file),
            os.path.join(train_dir, dog_name, file)
        )

    for file in test_files:
        shutil.copy(
            os.path.join(dog_path, file),
            os.path.join(test_dir, dog_name, file)
        )

    print(f"{dog_name}: {len(train_files)} train, {len(test_files)} test")

print("Done splitting data.")