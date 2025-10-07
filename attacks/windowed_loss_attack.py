from utils import calculate_loss, calculate_perplexity
from .base_attack import AttackBase


class WindowedLossAttack(AttackBase):
    def get_record_score(self, rec):
        """
        Given one record with `candidates` and `candidate_values`,
        compute ppx under both models and return the JSON-able dict.
        """
        text = rec["context"]
        candidate_value = rec["candidate"]
        candidate_position = (rec["gold_candidate_position"]["start"], rec["gold_candidate_position"]["start"] + len(candidate_value))
        ppx_contextualized = calculate_perplexity(
            self.trained_model, self.tokenizer, 
            texts=[text], section_spans=[candidate_position],
            continue_window=-1
        )
        return ppx_contextualized[0]
