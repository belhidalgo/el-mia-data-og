import json
import random
from download import compute_sha256

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def make_mia_files(true_trained, false_trained, false_untrained, K, seed=42):
    random.seed(seed)

    # Load all files into memory
    true_data = load_jsonl(true_trained)
    false_trained_data = load_jsonl(false_trained)
    false_untrained_data = load_jsonl(false_untrained)

    n = len(true_data)
    assert len(false_trained_data) >= n and len(false_untrained_data) >= n, \
        "Mismatch in file lengths!"

    # Pick K random indices
    indices = random.sample(range(n), K)

    mia_trained, mia_untrained, mia_mix = [], [], []

    for i in indices:
        # mia_trained
        mia_trained.append(true_data[i])
        mia_trained.append(false_trained_data[i])

        # mia_untrained
        mia_untrained.append(true_data[i])
        mia_untrained.append(false_untrained_data[i])

        # mia_mix
        mia_mix.append(true_data[i])
        if i % 2 == 1:  # odd
            mia_mix.append(false_trained_data[i])
        else:  # even
            mia_mix.append(false_untrained_data[i])

    # Write outputs
    write_jsonl("mia_trained.jsonl", mia_trained)
    write_jsonl("mia_untrained.jsonl", mia_untrained)
    write_jsonl("mia_mix.jsonl", mia_mix)

    print(f"Created mia_trained.jsonl, mia_untrained.jsonl, mia_mix.jsonl with {K*2} pairs each.")

make_mia_files(
    "true_trained.jsonl",
    "false_trained.jsonl",
    "false_untrained.jsonl",
    K=20_000   # half number of lines of target files
)

"""
SHA256 checksums:
mia_trained.jsonl: 94eb53335582c55841bf6e5baa129064d2f408be484f8e14b73075d8b62ba5d6
mia_untrained.jsonl: 05fd98c5caad61cbd8b4678857227459f155daec47705884a4a701f582f1d99a
mia_mix.jsonl: 8050ffacb934202783e5b040c25ceba8be466dade21f5c34cfcac2f07fde30be
"""

print("Verifying SHA256 checksum...")
file_hash1 = compute_sha256("mia_trained.jsonl")
file_hash2 = compute_sha256("mia_untrained.jsonl")
file_hash3 = compute_sha256("mia_mix.jsonl")

assert file_hash1 == "6e0532e3214f76956bafb67118544ba231861d285f8f810fdb7d150c0876bf5b", "❌ Checksum mismatch: mia_trained.jsonl"
assert file_hash2 == "58164375749673d07e948d62e0fee9d539b8e493de858b94a9c82c59943a341c", "❌ Checksum mismatch: mia_untrained.jsonl"
assert file_hash3 == "9c11796570a794dcc5f683eba5150f1022d2c7110dd4ecdd929066e48b3ca1a8", "❌ Checksum mismatch: mia_mix.jsonl"

print(f"✅ Checksum OK: {file_hash1}")
print(f"✅ Checksum OK: {file_hash2}")
print(f"✅ Checksum OK: {file_hash3}")