from collections import Counter

def count_duplicate_lines(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]  # strip to ignore trailing newlines/spaces
    counter = Counter(lines)

    # Only keep lines that appear more than once
    duplicates = {line: count for line, count in counter.items() if count > 1}

    print(f"Total duplicate lines: {len(duplicates)}")
    for line, count in duplicates.items():
        print(f"'{line}' appears {count} times")

    return duplicates

# Example usage
if __name__ == "__main__":
    filename = "mia_mix.jsonl"
    count_duplicate_lines(filename)
