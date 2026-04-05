import json
import random

INPUT_FILE = "dataset.json"
OUTPUT_FILE = "shuffled_dataset.json"
NUM_SAMPLES = 10

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

# Collect all description texts from the dataset
all_texts = []
for entry in data:
    for desc in entry["descriptions"]:
        all_texts.append(desc["text"])

all_texts = list(set(all_texts))  # remove duplicates globally

for entry in data:
    # Existing descriptions for this image
    existing_texts = set(desc["text"] for desc in entry["descriptions"])

    # Candidate incorrect descriptions must not already be used
    candidates = [t for t in all_texts if t not in existing_texts]

    # Randomly sample without duplicates
    num_to_sample = min(NUM_SAMPLES, len(candidates))
    sampled = random.sample(candidates, num_to_sample)

    # Add them as incorrect descriptions
    for text in sampled:
        entry["descriptions"].append({
            "text": text,
            "is_correct": False
        })

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=2)

print("Augmented dataset saved to", OUTPUT_FILE)