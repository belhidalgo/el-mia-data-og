#!/usr/bin/env python3
import json, argparse, random, sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# ---------- Helpers ----------

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: Path, obj_iter):
    with path.open("w", encoding="utf-8") as f:
        for obj in obj_iter:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_candidates(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect: { "LABEL": ["cand1", "cand2", ...], ... }
    for k, v in list(data.items()):
        if not isinstance(v, list):
            raise ValueError(f"Candidates for key '{k}' must be a list, got {type(v)}")
    return data

def fix_or_find_span(text: str, start: int, end: int, gold_value: str) -> Optional[Tuple[int, int]]:
    """
    Ensure (start, end) in 'text' correspond to 'gold_value'.
    If mismatch, try to find the first exact occurrence of 'gold_value' in 'text'.
    Return (start, end) or None if not found.
    """
    if 0 <= start <= end <= len(text) and text[start:end] == gold_value:
        return start, end
    # fallback: search
    idx = text.find(gold_value)
    if idx != -1:
        return idx, idx + len(gold_value)
    return None

def choose_negative(label: str, gold_value: str, pool: Dict[str, List[str]], rng: random.Random, max_tries: int = 100) -> Optional[str]:
    options = pool.get(label, [])
    if not options:
        return None
    # Try not-equal sampling
    for _ in range(min(max_tries, len(options) * 30)):
        cand = rng.choice(options)
        if cand != gold_value:
            return cand
    # If all options collapse to the same value as gold, give up
    print(f"Warning: all candidates for label '{label}' are identical to gold value '{gold_value}'.", file=sys.stderr)
    unique = set(options)
    if len(unique) == 1 and gold_value in unique:
        return None
    # As a final fallback, pick any other value if exists
    unique.discard(gold_value)
    return rng.choice(list(unique)) if unique else None

def make_record(context: str,
                gold_context: str,
                candidate: str,
                gold_candidate: str,
                cand_start: int,
                attribute: str,
                label_val: int,
                rec_id: Any) -> Dict[str, Any]:
    return {
        "context": context,
        "gold_context": gold_context,
        "candidate": candidate,
        "gold_candidate": gold_candidate,
        "gold_candidate_position": {"start": cand_start, "end": cand_start + len(candidate)},
        "attribute": attribute,
        "label": label_val,
        "id": rec_id,
    }

# ---------- Main pipeline ----------

def generate_sets(
    input_jsonl: Path,
    trained_cands_json: Path,
    untrained_cands_json: Path,
    out_true: Path,
    out_false_trained: Path,
    out_false_untrained: Path,
    seed: int,
):
    rng = random.Random(seed)
    trained_pool = load_candidates(trained_cands_json)
    untrained_pool = load_candidates(untrained_cands_json)

    true_out = out_true.open("w", encoding="utf-8")
    ftrain_out = out_false_trained.open("w", encoding="utf-8")
    funtrain_out = out_false_untrained.open("w", encoding="utf-8")

    # Simple stats
    n_input = n_spans = 0
    n_true = n_ftrain = n_futrain = 0
    skipped_bad_span = skipped_no_candidate_train = skipped_no_candidate_untrain = 0

    try:
        for rec in read_jsonl(input_jsonl):
            n_input += 1
            src = rec.get("source_text", "")
            rec_id = rec.get("id")
            spans = rec.get("privacy_mask", []) or []
            for s in spans:
                n_spans += 1
                gold_val = s["value"]
                start = int(s["start"])
                end = int(s["end"])
                label = s["label"]

                # Validate / repair span
                fixed = fix_or_find_span(src, start, end, gold_val)
                if not fixed:
                    skipped_bad_span += 1
                    continue
                start, end = fixed
                # ---------- TRUE (label=1) ----------
                true_record = make_record(
                    context=src,
                    gold_context=src,
                    candidate=gold_val,
                    gold_candidate=gold_val,
                    cand_start=start,
                    attribute=label,
                    label_val=1,
                    rec_id=rec_id,
                )
                true_out.write(json.dumps(true_record, ensure_ascii=False) + "\n")
                n_true += 1

                # ---------- FALSE_TRAINED (label=0) ----------
                neg_tr = choose_negative(label, gold_val, trained_pool, rng)
                if neg_tr is None:
                    skipped_no_candidate_train += 1
                else:
                    new_ctx = src[:start] + neg_tr + src[end:]
                    ftrain_record = make_record(
                        context=new_ctx,
                        gold_context=src,
                        candidate=neg_tr,
                        gold_candidate=gold_val,
                        cand_start=start,
                        attribute=label,
                        label_val=0,
                        rec_id=rec_id,
                    )
                    ftrain_out.write(json.dumps(ftrain_record, ensure_ascii=False) + "\n")
                    n_ftrain += 1

                # ---------- FALSE_UNTRAINED (label=0) ----------
                neg_untr = choose_negative(label, gold_val, untrained_pool, rng)
                if neg_untr is None:
                    skipped_no_candidate_untrain += 1
                else:
                    new_ctx2 = src[:start] + neg_untr + src[end:]
                    funtrain_record = make_record(
                        context=new_ctx2,
                        gold_context=src,
                        candidate=neg_untr,
                        gold_candidate=gold_val,
                        cand_start=start,
                        attribute=label,
                        label_val=0,
                        rec_id=rec_id,
                    )
                    funtrain_out.write(json.dumps(funtrain_record, ensure_ascii=False) + "\n")
                    n_futrain += 1
    finally:
        true_out.close()
        ftrain_out.close()
        funtrain_out.close()

    # Report
    print(f"Read {n_input} records, {n_spans} total spans.")
    print(f"Wrote: true={n_true}, false_trained={n_ftrain}, false_untrained={n_futrain}.")
    if skipped_bad_span:
        print(f"Skipped {skipped_bad_span} spans (bad/missing gold span in text).", file=sys.stderr)
    if skipped_no_candidate_train:
        print(f"Skipped {skipped_no_candidate_train} spans (no valid negative in candidates_trained).", file=sys.stderr)
    if skipped_no_candidate_untrain:
        print(f"Skipped {skipped_no_candidate_untrain} spans (no valid negative in candidates_untrained).", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Generate PII-MIA jsonl sets (true_trained / false_trained / false_untrained).")
    ap.add_argument("--input", type=Path, default=Path("train.jsonl"),
                    help="Input JSONL file with fields: source_text, privacy_mask, id, ...")
    ap.add_argument("--candidates_trained", type=Path, default=Path("candidates_trained.json"),
                    help="JSON with {label: [candidates...]}")
    ap.add_argument("--candidates_untrained", type=Path, default=Path("candidates_untrained.json"),
                    help="JSON with {label: [candidates...]}")
    ap.add_argument("--out_true", type=Path, default=Path("true_trained.jsonl"))
    ap.add_argument("--out_false_trained", type=Path, default=Path("false_trained.jsonl"))
    ap.add_argument("--out_false_untrained", type=Path, default=Path("false_untrained.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    generate_sets(
        input_jsonl=args.input,
        trained_cands_json=args.candidates_trained,
        untrained_cands_json=args.candidates_untrained,
        out_true=args.out_true,
        out_false_trained=args.out_false_trained,
        out_false_untrained=args.out_false_untrained,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
