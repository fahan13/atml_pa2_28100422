import torch
from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_tokenizer
from common.generation import response_sequence_logprobs
from common.logprobs import sequence_logprobs
from common.logging_utils import save_json
from task1_dpo.train import filter_rows, make_collate

cfg = load_yaml("configs/dpo.yaml")
tok = load_tokenizer(cfg["base_model"])
rows, _ = filter_rows(read_jsonl(cfg["paths"]["dpo_standard_eval"]), tok, 768)
batch, _ = make_collate(tok, 768)(rows[:2])
batch = {k: v.cuda() for k, v in batch.items()}
model = load_policy(cfg)
with torch.no_grad():
    released = response_sequence_logprobs(model, batch)[0]
    ours = sequence_logprobs(model, **batch)[0]
print("released:", released.tolist(), "\nours:    ", ours.tolist())
save_json("results/logprob_equivalence.json",
          {"released": released.tolist(), "ours": ours.tolist(),
           "max_abs_diff": float((released - ours).abs().max())})