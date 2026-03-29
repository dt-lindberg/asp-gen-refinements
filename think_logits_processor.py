"""ThinkLogitsProcessor — forces </think> once the thinking budget is exhausted.

Implements the vLLM v1 LogitsProcessor interface (vLLM >= 0.17.0).

Pass to LLM() as a FQCN string:
    LLM(..., logits_processors=["think_logits_processor:ThinkLogitsProcessor"])

vLLM will instantiate the class inside the EngineCore subprocess with:
    ThinkLogitsProcessor(vllm_config, device, is_pin_memory)

Design:
  - update_state() receives BatchUpdate with live references to each
    sequence's output_tok_ids list. These are mutable lists that vLLM
    updates in-place after every generated token, so apply() always
    sees the latest token history via the stored references.
  - apply() receives the full batch logits tensor [batch_size, vocab_size].
    For any sequence that has exceeded MAX_THINK_TOKENS without a </think>,
    the corresponding row is set to -inf except for THINK_END_TOKEN_ID.
"""

import torch
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from config import MAX_THINK_TOKENS, THINK_END_TOKEN_ID


class ThinkLogitsProcessor(LogitsProcessor):
    """Force </think> token once the per-sequence thinking budget is exhausted."""

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        self.device = device
        # index -> live output_tok_ids list (mutable reference updated by vLLM)
        self.active: dict[int, list[int]] = {}
        # index -> bool: True once </think> has been observed for that sequence
        self.think_ended: dict[int, bool] = {}

    def is_argmax_invariant(self) -> bool:
        # We change argmax when forcing </think> — not invariant
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if not batch_update:
            return

        # AddedRequest = (index, SamplingParams, prompt_tok_ids, output_tok_ids)
        for index, _params, _prompt, output_tok_ids in batch_update.added:
            self.active[index] = output_tok_ids
            self.think_ended[index] = False

        # RemovedRequest = int (batch index)
        for index in batch_update.removed:
            self.active.pop(index, None)
            self.think_ended.pop(index, None)

        # MovedRequest = (index_a, index_b, MoveDirectionality)
        for a_idx, b_idx, direction in batch_update.moved:
            a_toks = self.active.pop(a_idx, None)
            b_toks = self.active.pop(b_idx, None)
            a_ended = self.think_ended.pop(a_idx, False)
            b_ended = self.think_ended.pop(b_idx, False)
            if a_toks is not None:
                self.active[b_idx] = a_toks
                self.think_ended[b_idx] = a_ended
            if b_toks is not None and direction == MoveDirectionality.SWAP:
                self.active[a_idx] = b_toks
                self.think_ended[a_idx] = b_ended

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, output_tok_ids in self.active.items():
            # Already past </think> — do nothing
            if self.think_ended.get(index, False):
                continue

            # Check if </think> has just appeared in output
            if THINK_END_TOKEN_ID in output_tok_ids:
                self.think_ended[index] = True
                continue

            # Still within thinking budget — do nothing
            if len(output_tok_ids) < MAX_THINK_TOKENS:
                continue

            # Budget exhausted, no </think> yet — force it
            logits[index].fill_(float("-inf"))
            logits[index, THINK_END_TOKEN_ID] = 0.0

        return logits
