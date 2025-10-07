from utils import calculate_loss
from .base_attack import AttackBase


class LowestLossAttack(AttackBase):
    def get_record_score(self, rec):
        """
        Given one record with `candidates` and `candidate_values`,
        compute ppx under both models and return the JSON-able dict.
        """
        text = rec["context"]
        loss_tr = calculate_loss(self.trained_model,   self.tokenizer, [text])
        return loss_tr[0]
