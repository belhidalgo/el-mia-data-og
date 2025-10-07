from .perplexity_ratio_attack import PerplexityRatioAttack
from .min_k_prob_attack import MinKProbAttack
from .lowest_loss_attack import LowestLossAttack
from .loss_ratio_attack import LossRatioAttack
from .candidate_ratio_attack import LikelihoodRatioAttack
from .windowed_loss_attack import WindowedLossAttack

ATTACKS = {
    "perplexity_ratio": PerplexityRatioAttack,
    "lowest_ppx": LowestPpxAttack,
    "context_diff_abs": ContextDiffAbsAttack,
    "context_diff": ContextDiffAttack,
    "context_diff_norm": ContextDiffNormAttack,
    "multi_context": MultiContextAttack,
    "lowest_entropy": LowestEntropyAttack,
    "lowest_gradient": LowestGradientAttack,
    "min_k_prob": MinKProbAttack,
    "lowest_gradient_ratio": GradientRatioAttack,
    "lowest_loss": LowestLossAttack,
    "loss_ratio": LossRatioAttack,
    "ppx_entropy": PpxEntropyAttack,
    "windowed_loss": WindowedLossAttack,
    "candidate_ratio": LikelihoodRatioAttack
}