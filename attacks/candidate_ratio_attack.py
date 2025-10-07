from typing import Dict, List, Tuple, Optional
import math
import numpy as np

from utils import calculate_perplexity  # expects same signature as calculate_perplexity
from .base_attack import AttackBase
import random


def _logmeanexp(values: List[float]) -> float:
    """
    Numerically stable log-mean-exp over a list of log-values.
    Returns log( mean( exp(values) ) ).
    """
    if not values:
        return float("-inf")
    m = max(values)
    if math.isinf(m):
        return m
    s = sum(math.exp(v - m) for v in values)
    return m + math.log(s / len(values))


class LikelihoodRatioAttack(AttackBase):
    """
    Black-box Likelihood Ratio MIA using a windowed (section) loss.

    Given record `rec` with fields:
      - rec["context"]: full text containing the candidate
      - rec["candidate"]: candidate string E
      - rec["attribute"]: attribute type key (to select foils)
      - rec["gold_candidate_position"]["start"]: start char index of E in context

    Constructor expects:
      - reference_bank: Dict[str, List[str]] mapping attribute -> list of foils
      - num_refs: optional cap on number of foils to use (sampled without replacement)
      - score_type: "ratio" (default) or "z" to choose the primary score
      - continue_window: forwarded to loss function (e.g., -1 to only score the section)
    """

    def _build_ref_list(self, attr: str, candidate: str) -> List[str]:
        refs = self.reference_bank.get(attr, [])
        # Exclude the exact candidate (case-insensitive)
        refs = [r for r in refs if r != candidate]
        selected = []
        for _ in range(self.num_refs):
            selected.append(random.choice(refs))
        return selected

    def get_record_score(self, rec):
        """
        Compute the candidate's windowed loss and compare to foils in the same span
        (replacing the candidate text with each foil). Returns a dict with:
          - score: primary scalar (ratio or z)
          - ratio, z: both variants
          - ll_cand, ll_refs_mean, ll_refs_std, n_refs
        """
        text = rec["context"]
        candidate = rec["candidate"]
        attr = rec.get("attribute", None)

        # Locate candidate span
        start = rec["gold_candidate_position"]["start"]
        end = start + len(candidate)

        # Build reference foils for this attribute
        ref_list = self._build_ref_list(attr, candidate)
        # If no refs available, we still return a score based only on candidate loss
        # (ratio=+inf; z=NaN). This is useful for debugging edge cases.
        # Prepare batched texts + spans: first the original (candidate), then each foil variant
        texts = [text]
        spans = [(start, end)]

        # Create foil-ed contexts
        for foil in ref_list:
            new_text = text[:start] + foil + text[end:]
            new_span = (start, start + len(foil))
            texts.append(new_text)
            spans.append(new_span)

        # Compute section-average losses in one call (index-aligned with texts/spans)
        # Expected behavior: calculate_loss(model, tokenizer, texts=[...], section_spans=[...], continue_window=...)
        losses = calculate_perplexity(
            self.trained_model,
            self.tokenizer,
            texts=texts,
            section_spans=spans,
            continue_window=self.continue_window,
            return_loss=True
        )
        # losses[i] is average NLL over the span (per-token)
        loss_cand = float(losses[0])
        loss_refs = [float(x) for x in losses[1:]]  # may be empty

        # Convert to log-likelihoods: ll = -loss
        ll_cand = -loss_cand
        ll_refs = [-l for l in loss_refs]

        # Ratio score in probability space: ratio = p(E | S) / mean_i p(E'_i | S)
        # Compute in log-space: log_ratio = ll_cand - logmeanexp(ll_refs); ratio = exp(log_ratio)
        if ll_refs:
            log_ratio = ll_cand - _logmeanexp(ll_refs)
            ratio = float(math.exp(log_ratio))
            mean_ll = float(np.mean(ll_refs))
            std_ll = float(np.std(ll_refs))  # population std; for sample: ddof=1 (guard n>1)
            # z-score on log-likelihoods
            eps = 1e-12
            z = (ll_cand - mean_ll) / (std_ll + eps)
        else:
            # No refs: define sane fallbacks
            ratio = float("inf")
            z = float("nan")
            mean_ll = float("nan")
            std_ll = float("nan")

        # Primary score selection
        # primary = ratio if self.score_type == "ratio" else z

        return {
            "score": float(ratio),
            "ratio": float(ratio),
            "z": float(z),
            "ll_cand": float(ll_cand),
            "ll_refs_mean": float(mean_ll),
            "ll_refs_std": float(std_ll),
            "n_refs": int(len(ll_refs)),
            # For auditing/ablation:
            "loss_cand": float(loss_cand),
            "loss_refs_mean": float(np.mean(loss_refs)) if loss_refs else float("nan"),
        }
