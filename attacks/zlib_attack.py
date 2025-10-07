import argparse
import json
import zlib
from pathlib import Path
from glob import glob

def zlib_bits_per_char(text: str) -> float:
    if not text:
        return float("inf")
    raw = text.encode("utf-8", errors="ignore")
    comp = zlib.compress(raw, level=9)
    return (len(comp) * 8) / max(1, len(text))  # bits per *character*

def to_zlib_dir_name(dir_path: Path) -> Path:
    name = dir_path.name
    if name.endswith("__lowest_loss"):
        name = name[:-len("__lowest_loss")] + "__zlib"
    else:
        name = name + "__zlib"
    return dir_path.parent / name

def process_dir(dir_path: Path) -> None:
    src = dir_path / "results.jsonl"
    if not src.exists():
        print(f"[skip] {dir_path} (no results.jsonl)")
        return

    out_dir = to_zlib_dir_name(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "results.jsonl"

    n = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            ctx = obj.get("context", "")
            bpc = zlib_bits_per_char(ctx)

            # ---- SIMPLE NORMALIZATION: divide loss by zlib BPC ----
            # new_score = loss / zlib_bpc
            loss = float(obj["score"])
            obj["score"] = loss / bpc
            obj["zlib_bpc"] = bpc  # keep for transparency

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"[ok]  {dir_path.name} -> {out_dir.name}  ({n} lines)")

def main():
    ap = argparse.ArgumentParser(description="Normalize loss by zlib(BPC) and save to *__zlib/results.jsonl.")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dir", type=str, help="Single directory like outputs/mia/160m__1ep__mix__lowest_loss")
    grp.add_argument("--glob", type=str, help="Glob of directories, e.g. 'outputs/mia/*__lowest_loss'")
    args = ap.parse_args()

    if args.dir:
        process_dir(Path(args.dir))
    else:
        for p in sorted(glob(args.glob)):
            d = Path(p)
            if d.is_dir():
                process_dir(d)

if __name__ == "__main__":
    main()
