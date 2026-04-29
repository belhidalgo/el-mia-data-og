import json
from collections import defaultdict
import os
import random
import sys
from typing import Any, Dict, Optional, Tuple
from zipfile import Path
from dataset.generate_sentences import read_jsonl
from download import compute_sha256
from openai import OpenAI

# Input and output file paths
INPUT_FILE = 'raw.jsonl'
OUTPUT_FILE = 'candidates.json'
EXPECTED_SHA256 = "b5cba7654f037f6de953173b57c43999821e679bedc510d08066f8e2d82de38f"

def generate_paraphrase(words: list[tuple[str, str]]):
    """
    words: list of (value, label) pairs
    """
    openai = OpenAI()

    # Create a prompt for paraphrasing
    prompt = "Generate 5 different paraphrased simple sentences containing the following sensitive values on the same order:\n\n"
    for value, label in words:
        prompt += f"- {value} ({label})\n"
    prompt += "\nParaphrased values:\n"

    # Call OpenAI API to generate paraphrases
    response = openai.chat.completions.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates paraphrased sentences containing sensitive values."},
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

def fix_or_find_span(text: str, start: int, end: int, value: str) -> Optional[Tuple[int, int]]:
    """
    Ensure (start, end) in 'text' correspond to 'value'.
    If mismatch, try to find the first exact occurrence of 'value' in 'text'.
    Return (start, end) or None if not found.
    """
    if 0 <= start <= end <= len(text) and text[start:end] == value:
        return start, end
    # fallback: search
    idx = text.find(value)
    if idx != -1:
        return idx, idx + len(value)
    return None

def make_record(context: str,
                candidate: str,
                cand_start: int,
                attribute: str,
                rec_id: Any) -> Dict[str, Any]:
    return {
        "context": context,
        "candidate": candidate,
        "candidate_position": {"start": cand_start, "end": cand_start + len(candidate)},
        "attribute": attribute,
        "id": rec_id,
    }

def generate_sets(
    input_jsonl: Path,
    out_file: Path,
    seed: int,
):
    rng = random.Random(seed)

    out_file = out_file.open("w", encoding="utf-8")

    # Simple stats
    n_input = n_spans = 0
    skipped_bad_span = 0

    try:
        for rec in read_jsonl(input_jsonl):
            n_input += 1
            src = rec.get("source_text", "")
            rec_id = rec.get("id")
            spans = rec.get("privacy_mask", []) or []
            for s in spans:
                n_spans += 1
                val = s["value"]
                start = int(s["start"])
                end = int(s["end"])
                label = s["label"]

                # Validate / repair span
                fixed = fix_or_find_span(src, start, end, val)
                if not fixed:
                    skipped_bad_span += 1
                    continue
                start, end = fixed

                record = make_record(
                    context=src,
                    candidate=val,
                    cand_start=start,
                    attribute=label,
                    rec_id=rec_id,
                )
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        out_file.close()

    # Report
    print(f"Read {n_input} records, {n_spans} total spans.")
    if skipped_bad_span:
        print(f"Skipped {skipped_bad_span} spans (bad/missing gold span in text).", file=sys.stderr)

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