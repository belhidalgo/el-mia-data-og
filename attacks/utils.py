import argparse
import json
import os
import random
import shutil
from typing import List, Optional, Tuple
import numpy as np
import torch
import subprocess
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from datasets import Dataset, load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def load_data(args):
    """
    Loads data from text files (one line = one sample).
    Returns two Dataset objects: train_dataset, val_dataset.
    """
    if args.train_file.endswith('.jsonl'):
        train_data = load_dataset("json", data_files=args.train_file, split="train")
        val_data = load_dataset("json", data_files=args.validation_file, split="train")
        eval_data = load_dataset("json", data_files=args.eval_file, split="train")
        
        # Extract the 'source_text' field into lists
        train_lines = [item["source_text"] for item in train_data]
        val_lines = [item["source_text"] for item in val_data]
        eval_lines = [item["source_text"] for item in eval_data]

    else:
        # Read train text file line-by-line
        with open(args.train_file, 'r', encoding='utf-8') as f:
            train_lines = [line.strip() for line in f if line.strip()]
        # Read validation text file line-by-line
        with open(args.validation_file, 'r', encoding='utf-8') as f:
            val_lines = [line.strip() for line in f if line.strip()]        
        # Read eval text file line-by-line
        with open(args.eval_file, 'r', encoding='utf-8') as f:
            eval_lines = [line.strip() for line in f if line.strip()]
    
    # Create Dataset objects from these lines
    # Each line is an entry in the "text" column
    train_dataset = Dataset.from_dict({"text": train_lines})
    val_dataset = Dataset.from_dict({"text": val_lines})
    eval_dataset = Dataset.from_dict({"text": eval_lines})
    
    return train_dataset, val_dataset, eval_dataset


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=True)


def generate_branch_name(epoch):
    return f"epoch-{epoch+1}"


def generate_load_branch_name(epoch):
    return f"epoch-{epoch}"


def clear_directory(dir_path):
    """
    Removes all files and subdirectories from `dir_path`,
    """
    # Recursively delete the entire directory
    shutil.rmtree(dir_path, ignore_errors=True)


def load_data_from_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records


import os, time
import torch.distributed as dist

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    # torchrun sets RANK; fall back to SLURM_PROCID
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))) == 0

# after args = parse_args()
def safe_makedirs(dir_path):
    """
    Creates the directory if it doesn't exist.
    """
    if is_main_process():
        os.makedirs(dir_path, exist_ok=True)

# don't barrier unless process group is up
if is_dist_avail_and_initialized():
    dist.barrier()


from typing import List, Optional
import torch
from torch.nn import CrossEntropyLoss

def calculate_loss(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 4,
    add_start_token: bool = True
) -> List[float]:
    """
    Calculates the average token-level loss (cross-entropy) for each input text.
    Returns a list of loss values, one per input.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    loss_fct = CrossEntropyLoss(reduction="none")

    # Tokenize all inputs
    encodings = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True
    ).to(device)

    input_ids = encodings["input_ids"]
    attn_mask = encodings["attention_mask"]

    losses = []
    for start in range(0, len(input_ids), batch_size):
        end = min(start + batch_size, len(input_ids))
        batch_ids  = input_ids[start:end]
        batch_mask = attn_mask[start:end]

        # Optionally prepend BOS token
        if add_start_token and tokenizer.bos_token_id is not None:
            bos = torch.full((batch_ids.size(0), 1), tokenizer.bos_token_id, device=device)
            batch_ids  = torch.cat([bos, batch_ids], dim=1)
            batch_mask = torch.cat([torch.ones_like(bos), batch_mask], dim=1)

        with torch.no_grad():
            logits = model(batch_ids, attention_mask=batch_mask).logits

        # Prepare shifted inputs for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_ids[..., 1:].contiguous()
        shift_mask   = batch_mask[..., 1:].contiguous()

        # Compute token-level loss
        token_loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)  # (B, L−1)

        # Masked average loss per sequence
        avg_loss = (token_loss * shift_mask).sum(1) / shift_mask.sum(1)
        losses.extend(avg_loss.tolist())

    return losses


def calculate_perplexity(
    model,
    tokenizer,
    texts: List[str],
    section_spans: Optional[List[Optional[Tuple[int, int]]]] = None,  # [(start_char, end_char)] per text
    batch_size: int = 4,
    add_start_token: bool = True,
    continue_window: Optional[int] = None,
    return_loss: bool = False
) -> List[float]:
    """
    Calculates the (average) token perplexity of each text in `texts`.

    If `section_spans` is provided (same length as `texts`), it must be a list
    where each item is either:
      - None: use the full valid range of tokens for that text
      - (char_start, char_end): a half-open character span [start, end) in the original text
    Only tokens whose character offsets intersect that span will be used for the loss.

    continue_window:
      - None: mask exactly the tokens overlapping the span
      - int n >= 0: extend the mask n tokens after the span (within valid range)
      - -1: extend the mask to the end of the valid (unmasked) sequence
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    loss_fct = CrossEntropyLoss(reduction="none")

    if section_spans is not None and len(section_spans) != len(texts):
        raise ValueError("`section_spans` must be the same length as `texts`")

    # Tokenize for model inputs (tensors)
    enc_pt = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = enc_pt["input_ids"].to(device)
    attn_mask = enc_pt["attention_mask"].to(device)

    # Tokenize again to get offset mappings (lists of (start,end) char spans)
    enc_off = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    )
    # enc_off["offset_mapping"]: List[List[Tuple[int,int]]] including padding offsets
    losses = []
    ppls = []
    for start in range(0, len(input_ids), batch_size):
        end = min(start + batch_size, len(input_ids))
        batch_ids  = input_ids[start:end].clone()
        batch_mask = attn_mask[start:end].clone()

        # Optionally prepend BOS
        if add_start_token and tokenizer.bos_token_id is not None:
            bos = torch.tensor([[tokenizer.bos_token_id]] * batch_ids.size(0), device=device)
            batch_ids  = torch.cat([bos, batch_ids], dim=1)
            batch_mask = torch.cat([torch.ones_like(bos, dtype=batch_mask.dtype), batch_mask], dim=1)

        with torch.no_grad():
            logits = model(batch_ids, attention_mask=batch_mask).logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()   # (B, L-1, V)
        shift_labels = batch_ids[..., 1:].contiguous()    # (B, L-1)
        shift_mask   = batch_mask[..., 1:].contiguous()   # (B, L-1)

        # Token-wise loss
        token_loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)  # (B, L-1)

        # Build selection mask from char spans if provided
        if section_spans is not None:
            sec_mask = torch.zeros_like(shift_mask)

            for i in range(end - start):
                # Default: use all valid tokens if span is None
                span = section_spans[start + i] if section_spans is not None else None
                if span is None:
                    sec_mask[i] = shift_mask[i]
                    continue

                char_start, char_end = span
                # Validate span; fall back to full mask if bad
                if not isinstance(char_start, int) or not isinstance(char_end, int) or char_start < 0 or char_end <= char_start:
                    sec_mask[i] = shift_mask[i]
                    continue

                offsets = enc_off["offset_mapping"][start + i]  # token-level (start_char, end_char) offsets
                # valid_len = number of valid positions in the *shifted* sequence
                valid_len = int(shift_mask[i].sum().item())

                # Collect token indices (unshifted indices) whose char offsets overlap [char_start, char_end)
                token_indices = []
                for t, (s_char, e_char) in enumerate(offsets):
                    # Padding or empty tokens usually have (0,0)
                    if s_char == e_char == 0:
                        continue
                    # overlap test
                    if not (e_char <= char_start or s_char >= char_end):
                        token_indices.append(t)

                # Convert unshifted token indices to shifted positions (t -> t-1)
                shift_indices = [t - 1 for t in token_indices if 1 <= t <= valid_len]

                if len(shift_indices) == 0:
                    # If nothing overlapped (e.g., span only hit the first token that gets dropped),
                    # fallback to first valid token onward or full mask (choose behavior).
                    # Here we fallback to full mask to avoid NaNs.
                    sec_mask[i] = shift_mask[i]
                else:
                    # Apply continue_window logic
                    if continue_window is not None:
                        start_pos = min(shift_indices)
                        if continue_window == -1:
                            end_pos = valid_len
                        else:
                            # Extend by length of matched region + continue_window
                            end_pos = min(max(shift_indices) + 1 + continue_window, valid_len)
                        sec_mask[i, start_pos:end_pos] = 1
                    else:
                        for idx in shift_indices:
                            if 0 <= idx < valid_len:
                                sec_mask[i, idx] = 1

                # Safety: if mask ended up empty, use full valid tokens
                if sec_mask[i].sum().item() == 0:
                    sec_mask[i] = shift_mask[i]
            
            loss = (token_loss * sec_mask).sum(1) / sec_mask.sum(1)
        else:
            # Full-sequence loss
            loss = (token_loss * shift_mask).sum(1) / shift_mask.sum(1)

        losses.extend(loss.tolist())
        ppls.extend(torch.exp(loss).tolist())
        
    if return_loss:
        return losses
    else:
        return ppls



def calculate_perplexity_with_entropy(
    model,
    tokenizer,
    texts: List[str],
    sections: Optional[List[Optional[str]]] = None,
    batch_size: int = 4,
    add_start_token: bool = True
) -> Tuple[
      List[float],             # avg_ppls
      List[float],             # avg_ents
      List[List[float]],       # section_or_full_token_ppls
      List[List[float]]        # section_or_full_entropies
]:
    """
    Returns:
      - avg_ppls: average (section‐masked) perplexity per text
      - avg_ents: average (section‐masked) entropy per text
      - section_token_ppls: per‐token perplexities *only* for the section (or full text if no section)
      - section_entropies:  per‐token entropies  *only* for the section (or full text if no section)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    loss_fct = CrossEntropyLoss(reduction="none")

    # Pre-tokenize section strings
    if sections is not None:
        if len(sections) != len(texts):
            raise ValueError("`sections` must match `texts` in length")
        section_token_ids = [
            tokenizer(s, add_special_tokens=False).input_ids if s else []
            for s in sections
        ]
    else:
        section_token_ids = [None] * len(texts)

    # Tokenize all texts
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True
    ).to(device)
    input_ids  = enc["input_ids"]       # (N, L)
    attn_mask  = enc["attention_mask"]  # (N, L)

    avg_ppls    = []
    avg_ents    = []
    sec_ppls    = []  # per‐example list of section‐only token ppls
    sec_ents    = []  # per‐example list of section‐only token entropies

    for i in range(0, len(input_ids), batch_size):
        batch_ids  = input_ids[i : i+batch_size]
        batch_mask = attn_mask[i : i+batch_size]
        batch_secs = section_token_ids[i : i+batch_size]

        # Optionally prepend BOS
        if add_start_token and tokenizer.bos_token_id is not None:
            bos = torch.full(
                (batch_ids.size(0), 1),
                tokenizer.bos_token_id,
                dtype=batch_ids.dtype,
                device=device
            )
            batch_ids  = torch.cat([bos, batch_ids],   dim=1)
            batch_mask = torch.cat([torch.ones_like(bos), batch_mask], dim=1)

        with torch.no_grad():
            logits = model(batch_ids, attention_mask=batch_mask).logits
        # shift for next-token
        shift_logits = logits[..., :-1, :].contiguous()  # (B, L, V)
        shift_labels = batch_ids[..., 1:].contiguous()   # (B, L)
        shift_mask   = batch_mask[..., 1:].contiguous()  # (B, L)

        # token-wise loss, ppls, entropies
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view_as(shift_labels)                         # (B, L)
        token_ppls  = torch.exp(token_loss)             # (B, L)
        probs       = F.softmax(shift_logits, dim=-1)   # (B, L, V)
        epsilon = 1e-12
        entropies = -(probs * (probs + epsilon).log()).sum(dim=-1)  # (B, L)

        # Build section mask same as before
        sec_mask = torch.zeros_like(shift_mask)
        for b, sec_ids in enumerate(batch_secs):
            if sec_ids:
                seq = batch_ids[b,1:]   # aligned with shift_labels
                m = len(sec_ids)
                found = False
                for pos in range(seq.size(0)-m+1):
                    if torch.equal(seq[pos:pos+m], torch.tensor(sec_ids, device=seq.device)):
                        sec_mask[b, pos:pos+m] = 1
                        found = True
                        break
                if not found:
                    # fallback: use full mask
                    sec_mask[b] = shift_mask[b]
            else:
                # no section provided → mask = full sequence
                sec_mask[b] = shift_mask[b]

        # avg perplexity over section tokens
        avg_loss = (token_loss * sec_mask).sum(dim=1) / sec_mask.sum(dim=1)
        avg_ppls.extend(torch.exp(avg_loss).tolist())
        avg_ents.extend(entropies.mean(dim=1).tolist())

        # collect section-only token ppls and entropies
        for b in range(token_ppls.size(0)):
            mask_inds = sec_mask[b].nonzero(as_tuple=False).view(-1).tolist()
            sec_ppls.append([token_ppls[b, idx].item() for idx in mask_inds])
            sec_ents.append([entropies[b,   idx].item() for idx in mask_inds])

    return avg_ppls, avg_ents, sec_ppls, sec_ents



if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Use the correct model name for Pythia-14M (double-check the HuggingFace model hub for the exact name)
    model_name = "EleutherAI/pythia-14m"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Two random sentences
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Intelligence is transforming the world."
    ]

    # Call the function without sections
    # avg_ppls, avg_ents, sec_ppls, sec_ents = calculate_perplexity_with_entropy(
    #     model, tokenizer, texts, sections=None
    # )

    # print("Average Perplexities:", avg_ppls)
    # print("Average Entropies:", avg_ents)
    # print("Section/Full Token Perplexities:", sec_ppls)
    # print("Section/Full Token Entropies:", sec_ents)
    losses = calculate_loss(model, tokenizer, texts)
    print("Losses:", losses)