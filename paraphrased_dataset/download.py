"""
1. Download English subset from https://huggingface.co/datasets/ai4privacy/pii-masking-200k


"""


import hashlib
import requests
import os

URL = "https://huggingface.co/datasets/ai4privacy/pii-masking-200k/resolve/main/english_pii_43k.jsonl"
OUTPUT_FILE = "raw.jsonl"
EXPECTED_SHA256 = "1a7aac12051b87390a18fe8ece2010cfa8c1631e0b69c46e6ed5a995b374a7f8"


def download_file(url, output_path):
    """Download a file from URL and save it locally."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def compute_sha256(file_path):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def run_download():
    # Download file
    print(f"Downloading from {URL} ...")
    download_file(URL, OUTPUT_FILE)

    # Verify checksum
    print("Verifying SHA256 checksum...")
    file_hash = compute_sha256(OUTPUT_FILE)
    if file_hash == EXPECTED_SHA256:
        print(f"✅ Checksum OK: {file_hash}")
    else:
        print(f"❌ Checksum mismatch!\nExpected: {EXPECTED_SHA256}\nGot:      {file_hash}")
        # Remove file if invalid
        os.remove(OUTPUT_FILE)
        raise ValueError("File checksum mismatch. Deleted corrupted file.")