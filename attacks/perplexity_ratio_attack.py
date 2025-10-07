import json
from tqdm import tqdm

from utils import calculate_perplexity
from .base_attack import AttackBase

import torch
from transformers import AutoModelForCausalLM

class PerplexityRatioAttack(AttackBase):
    def __init__(self, args):
        super().__init__(args)
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_mia(self):
        """Two-pass MIA:
           1) compute all ppx_tr with trained model on GPU,
           2) compute all ppx_un with untrained model on GPU,
           3) write results with ratio ppx_tr/ppx_un.
        """
        import json
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ===== PASS 1: trained model =====
        self.trained_model.to(device)
        ppx_tr_list = []
        with open(self.args.eval_file, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc="Pass 1: trained model (ppx_tr)"):
                rec = json.loads(line)
                text = rec["context"]
                ppx_tr = calculate_perplexity(self.trained_model, self.tokenizer, [text])[0]
                ppx_tr_list.append(ppx_tr)

        # offload trained to free GPU
        self.trained_model.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()
        del self.trained_model

        # ===== PASS 2: untrained model =====
        self.untrained_model = AutoModelForCausalLM.from_pretrained(self.args.model_name)
        self.untrained_model.to(device)
        with open(self.args.eval_file, "r", encoding="utf-8") as fin, \
             open(self.results_path, "w", encoding="utf-8") as fout:
            for i, line in enumerate(tqdm(fin, desc="Pass 2: untrained model (ppx_un)")):
                rec = json.loads(line)
                text = rec["context"]
                ppx_un = calculate_perplexity(self.untrained_model, self.tokenizer, [text])[0]
                rec["score"] = ppx_tr_list[i] / ppx_un
                fout.write(json.dumps(rec) + "\n")

        # tidy up
        self.untrained_model.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()

        print("Inference completed. Results saved.")

    def get_record_score(self, rec):
        """
        Given one record with `candidates` and `candidate_values`,
        compute ppx under both models and return the JSON-able dict.
        """
        # text = rec["context"]
        # candidate_value = rec["candidate"]
        # ppx_tr = calculate_perplexity(self.trained_model,   self.tokenizer, [text])
        # ppx_un = calculate_perplexity(self.untrained_model, self.tokenizer, [text])
        # return ppx_tr[0]/ppx_un[0]
        raise NotImplementedError