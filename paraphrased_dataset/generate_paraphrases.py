import json
from collections import defaultdict
import os
import random
from download import compute_sha256
from openai import OpenAI

# Input and output file paths
INPUT_FILE = 'raw.jsonl'
OUTPUT_FILE = 'candidates.json'
EXPECTED_SHA256 = "b5cba7654f037f6de953173b57c43999821e679bedc510d08066f8e2d82de38f"
TRAIN_SPLIT_LINES = 20_000  # Number of lines that has been trained on model training split
# Save the first TRAIN_SPLIT_LINES lines to a new file
TRAIN_SPLIT_FILE = 'train.jsonl'

def generate_paraphrase(words: list[tuple[str, str]]):
    """
    words: list of (value, label) pairs
    """
    openai = OpenAI()

    # Create a prompt for paraphrasing
    prompt = "Generate 5 different paraphrased simple sentences containing the following sensitive values:\n\n"
    for value, label in words:
        prompt += f"- {value} ({label})\n"
    prompt += "\nParaphrased values:\n"

    # Call OpenAI API to generate paraphrases
    response = openai.chat.completions.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates logical paraphrased sentences based on sensitive values while preserving their meaning."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000,
        n=5,
        stop=None,
    )

    # Extract paraphrased values from the response
    paraphrased_text = response.choices[0].message.content.strip()
    paraphrased_lines = paraphrased_text.split("\n")
    
    paraphrased_values = []
    for line in paraphrased_lines:
        if line.startswith("- "):
            try:
                value_label = line[2:].rsplit("(", 1)
                if len(value_label) == 2:
                    value = value_label[0].strip()
                    label = value_label[1].rstrip(")").strip()
                    paraphrased_values.append((value, label))
            except Exception as e:
                print(f"Error parsing line: {line}\n{e}")
    
    return paraphrased_values


def run_generate_paraphrases():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        candidates = defaultdict(list)
        for i, line in enumerate(f):
            item = json.loads(line)
            for entity in item['privacy_mask']:
                label = entity['label']
                value = entity['value']
                candidates[i].append((value, label))
        paraphrased = generate_paraphrase(candidates)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(paraphrased, f, indent=2)

    print(f"Saved sensitive candidates to {OUTPUT_FILE}")

    # Verify checksum
    print("Verifying SHA256 checksum...")
    file_hash = compute_sha256(OUTPUT_FILE)

    print(file_hash)

    if file_hash == EXPECTED_SHA256:
        print(f"✅ Checksum OK: {file_hash}")
    else:
        print(f"❌ Checksum mismatch!")
        # Remove file if invalid
        # os.remove(OUTPUT_TRAINED_FILE)
        raise ValueError("File checksum mismatch. Deleted corrupted file.")