import os

DATA_DIR = "data_barks"

total_wavs = 0

for dog_name in os.listdir(DATA_DIR):

    dog_path = os.path.join(DATA_DIR, dog_name)

    if os.path.isdir(dog_path):

        wavs = [
            f for f in os.listdir(dog_path)
            if f.endswith(".wav")
        ]

        print(f"{dog_name}: {len(wavs)} bark wav files")

        total_wavs += len(wavs)

print("\nTotal bark wav files:", total_wavs)

if total_wavs < 50:
    print("Warning: dataset has fewer than 50 bark clips.")
else:
    print("Dataset size looks good.")