"""Memory-efficient response-token log-probabilities (shared by Tasks 1-3).

The released helpers in common/generation.py materialize a float32 log-softmax over the full
vocabulary (152k) for every position, which does not fit an 8 GB GPU during training. This
version computes the SAME quantity (sum of response-token log-probs under teacher forcing) but
(i) only keeps logits for the response region via `logits_to_keep`, and (ii) uses
cross_entropy, which never materializes the full log-softmax.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_logprobs(model, input_ids, attention_mask, response_mask):
    """Return (sum over response tokens, per-token logp, per-token mask).

    Works for any layout where all response tokens lie at or after a common column
    (left-padded DPO batches, and generated sequences [pad|prompt|response|pad]).
    """
    any_resp = response_mask.bool().any(0)
    first = int(any_resp.nonzero()[0].item())          # first column holding a response token
    seq_len = input_ids.shape[1]
    keep = seq_len - first + 1                          # +1: logits at t-1 predict token t

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=keep,
        return_dict=True,
    )
    logits = out.logits                                 # [B, keep, V]
    B, K, V = logits.shape
    # logits[:, j] predicts token at column first + j. Last logit has no target -> dummy label, masked.
    labels = torch.cat([input_ids[:, first:], input_ids[:, -1:]], dim=1)
    mask = torch.cat([response_mask[:, first:], torch.zeros_like(response_mask[:, :1])], dim=1).float()
    tok = -F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="none").view(B, K).float()
    tok, mask = tok[:, :-1] * mask[:, :-1], mask[:, :-1]
    return tok.sum(-1), tok, mask