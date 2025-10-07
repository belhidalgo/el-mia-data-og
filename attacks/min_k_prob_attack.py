import math
from utils import calculate_perplexity_with_entropy
from .base_attack import AttackBase


class MinKProbAttack(AttackBase):
    def __init__(self, args):
        """
        Args:
            k_percent (int): percentage of lowest log-prob tokens to average.
        """
        super().__init__(args)
        self.k_percent = args.k_percent
        
    def get_record_score(self, rec):
        """
        Given one record with `candidates` and `candidate_values`,
        compute ppx under both models and return the JSON-able dict.
        """
        text = rec["context"]
        candidate_value = rec["candidate"]
        avg_ppls, avg_ents, token_ppls, token_entropies = calculate_perplexity_with_entropy(
            model=self.trained_model,
            tokenizer=self.tokenizer,
            texts=[text]
        )
        token_ppl_list = token_ppls[0]
        if len(token_ppl_list) == 0:
            avg_min_k_logprob = float('-inf')  # fail-safe for empty
        else:
            log_probs = [-math.log(ppl) for ppl in token_ppl_list]  # convert to log-probs
            k = max(1, int(len(log_probs) * self.k_percent / 100))
            min_k_log_probs = sorted(log_probs)[:k]  # lowest log-probs (least confident)
            avg_min_k_logprob = sum(min_k_log_probs) / k

        return avg_min_k_logprob
