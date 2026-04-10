"""Diagnostic variant of ThinkLogitsProcessor.

Reads the thinking budget from the DIAG_THINK_BUDGET environment variable
(default: 30) so it can be set per-job without touching config.py.

Also prints a message every time it forces </think>, so the action is
visible in the SLURM output log.

Pass to LLM() as:
    logits_processors=["diagnose_think_processor:DiagnoseThinkProcessor"]
"""

import os

import torch
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from config import THINK_END_TOKEN_ID

DIAG_THINK_BUDGET = int(os.environ.get("DIAG_THINK_BUDGET", "30"))
print(
    f"[DiagnoseThinkProcessor] loaded — DIAG_THINK_BUDGET={DIAG_THINK_BUDGET}, "
    f"THINK_END_TOKEN_ID={THINK_END_TOKEN_ID}",
    flush=True,
)


class DiagnoseThinkProcessor(LogitsProcessor):
    """Force </think> once DIAG_THINK_BUDGET output tokens have been generated."""

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.active: dict[int, list[int]] = {}
        self.think_ended: dict[int, bool] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if not batch_update:
            return

        for index, _params, _prompt, output_tok_ids in batch_update.added:
            self.active[index] = output_tok_ids
            self.think_ended[index] = False

        for index in batch_update.removed:
            self.active.pop(index, None)
            self.think_ended.pop(index, None)

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
            if self.think_ended.get(index, False):
                continue

            if THINK_END_TOKEN_ID in output_tok_ids:
                self.think_ended[index] = True
                continue

            if len(output_tok_ids) < DIAG_THINK_BUDGET:
                continue

            # Budget exhausted — force </think> and report it
            print(
                f"[DiagnoseThinkProcessor] FORCING </think> at output_tok_ids "
                f"len={len(output_tok_ids)} for batch index {index}",
                flush=True,
            )
            logits[index].fill_(float("-inf"))
            logits[index, THINK_END_TOKEN_ID] = 0.0

        return logits
