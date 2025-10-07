import os
import subprocess
import sys

scripts = [
    "download.py",
    "generate_candidates.py",
    "generate_sentences.py",
    "build_mia_datasets.py"
]

if __name__ == "__main__":
    for script in scripts:
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"{script} failed.")
            sys.exit(result.returncode)

    files_to_delete = [
        "candidates_trained.json",
        "candidates_untrained.json",
        "false_trained.jsonl",
        "false_untrained.jsonl",
        "raw.jsonl",
        "train.jsonl",
        "true_trained.jsonl"
    ]
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except FileNotFoundError:
            print(f"{file_path} not found.")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            
    # then make a directory called data and move "mia_trained.jsonl", "mia_untrained.jsonl", "mia_mix.jsonl" into it
    os.makedirs("data", exist_ok=True)
    for file_name in ["mia_trained.jsonl", "mia_untrained.jsonl", "mia_mix.jsonl"]:
        try:
            os.rename(file_name, os.path.join("data", file_name))
            print(f"Moved {file_name} to data/")
        except FileNotFoundError:
            print(f"{file_name} not found.")
        except Exception as e:
            print(f"Error moving {file_name}: {e}")
            
    print()
    print("✅ Datasets are ready in the 'data' directory.")