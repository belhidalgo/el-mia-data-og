import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import utils
from evaluation import evaluate_mia, evaluate_ranked_results
from utils import load_data_from_jsonl
from tabulate import tabulate

class AttackBase:
    def __init__(self, args):
        self.args = args
        self.results_path = os.path.join(args.output_dir, "results.jsonl")
        self.overall_csv = os.path.join(args.output_dir, "overall_mia_performance.csv")
        self.attr_csv = os.path.join(args.output_dir, "attribute_mia_performance.csv")
        self.mia_csv = os.path.join(args.output_dir, "mia_performance.csv")

        self.continue_window = args.continue_window
        self.num_refs = args.num_refs
        self.reference_bank = json.load(open(args.reference_bank, "r", encoding="utf-8")) if args.reference_bank else {}

        """ Load a specific model checkpoint and run evaluation. """
        print(f"Loading model from Hugging Face: {args.hf_username}/{args.hf_model_name}")
        print(f"Using checkpoint: checkpoint-{args.epoch_to_load}-epochs")


        # Load dataset
        # train_dataset, val_dataset, eval_dataset = utils.load_data(args)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load trained model from Hugging Face
        model_repo = f"{args.hf_username}/{args.hf_model_name}"

        if args.epoch_to_load == 0:
            self.trained_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        else:
            checkpoint_revision = utils.generate_load_branch_name(args.epoch_to_load)
            self.trained_model = AutoModelForCausalLM.from_pretrained(model_repo, revision=checkpoint_revision).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_repo, revision=checkpoint_revision)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def run_mia(self):
        """Run the attack and return results as a list/dict."""
        with open(self.args.eval_file,  "r", encoding="utf-8") as fin, \
             open(self.results_path, "w", encoding="utf-8") as fout:
            # Skip lines up to start_index
            for _ in range(getattr(self.args, 'start_index', 0)):
                next(fin, None)
            for line in tqdm(fin, desc="Running inference"):
                rec = json.loads(line)
                result = self.get_record_score(rec)
                if isinstance(result, dict):
                    rec.update(result)
                else:
                    rec["score"] = result
                fout.write(json.dumps(rec) + "\n")
        print("Inference completed. Results saved.")
    
    def run_inference_on_record(self, rec):
        raise NotImplementedError
    
    def get_record_score(self, rec):
        raise NotImplementedError

    def aggregate_mia_scores(self, threshold=None):
        """single data point level"""
        records = load_data_from_jsonl(self.results_path)
        results_df = evaluate_mia(records)
    
        # Save CSVs
        results_df.to_csv(self.mia_csv, index=False)

        # Optional: print summary
        print("\n=== MIA Performance ===")
        print(tabulate(results_df, headers="keys", tablefmt="pretty", showindex=False))
        print("")
        return results_df
    