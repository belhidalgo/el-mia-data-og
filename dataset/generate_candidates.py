import json
from collections import defaultdict
import os
import random
from download import compute_sha256

# Input and output file paths
INPUT_FILE = 'raw.jsonl'
OUTPUT_TRAINED_FILE = 'candidates_trained.json'
OUTPUT_UNTRAINED_FILE = 'candidates_untrained.json'
EXPECTED_SHA256_TRAINED = "468200980b38cea9f3a91cf5cc9450392305c7cd61619cb305eb089d974dccb7"
EXPECTED_SHA256_UNTRAINED = "79d39b3ad0d7f63c3a8acbb14648d193e91cf532729ea6bba04ffbadd4841cf9"
TRAIN_SPLIT_LINES = 20_000  # Number of lines that has been trained on model training split
# Save the first TRAIN_SPLIT_LINES lines to a new file
TRAIN_SPLIT_FILE = 'train.jsonl'

# A dictionary to store category-wise candidates
candidates_trained = defaultdict(set)
candidates_untrained = defaultdict(set)

with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(TRAIN_SPLIT_FILE, 'w', encoding='utf-8') as fout:
    for i, line in enumerate(fin):
        if i < TRAIN_SPLIT_LINES:
            fout.write(line)
        else:
            break

# Read and parse each line in the jsonl file
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        for entity in item['privacy_mask']:
            label = entity['label']
            value = entity['value']
            if i < TRAIN_SPLIT_LINES:
                candidates_trained[label].add(value)
            else:
                candidates_untrained[label].add(value)

# Convert sets to lists for JSON serializability
# but sort lists first
# then shuffle with a specific seed for reproducibility
random.seed(42)
candidates_trained = {label: random.sample(sorted(list(values)), len(values)) for label, values in candidates_trained.items()}
candidates_untrained = {label: random.sample(sorted(list(values)), len(values)) for label, values in candidates_untrained.items()}

# Save to JSON
with open(OUTPUT_TRAINED_FILE, 'w', encoding='utf-8') as f:
    json.dump(candidates_trained, f, indent=2)

print(f"Saved trained sensitive candidates to {OUTPUT_TRAINED_FILE}")

with open(OUTPUT_UNTRAINED_FILE, 'w', encoding='utf-8') as f:
    json.dump(candidates_untrained, f, indent=2)
print(f"Saved untrained sensitive candidates to {OUTPUT_UNTRAINED_FILE}")

# Verify checksum
print("Verifying SHA256 checksum...")
file_hash1 = compute_sha256(OUTPUT_TRAINED_FILE)
file_hash2 = compute_sha256(OUTPUT_UNTRAINED_FILE)

print(file_hash1)
print(file_hash2)

if file_hash1 == EXPECTED_SHA256_TRAINED and file_hash2 == EXPECTED_SHA256_UNTRAINED:
    print(f"✅ Checksum OK: {file_hash1}")
    print(f"✅ Checksum OK: {file_hash2}")
else:
    print(f"❌ Checksum mismatch!")
    # Remove file if invalid
    # os.remove(OUTPUT_TRAINED_FILE)
    raise ValueError("File checksum mismatch. Deleted corrupted file.")
