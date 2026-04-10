"""One-off script to find the token IDs for <think> and </think> in the Qwen3 tokenizer.

Run via sbatch (find_think_tokens.job) on a GPU node. Hard-code the printed IDs
into config.py as THINK_START_TOKEN_ID and THINK_END_TOKEN_ID.
"""

import sys

from config import (
    MODEL_PATH,
    MAX_MODEL_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    GPU_MEMORY_UTILIZATION,
    SEED,
)

print(f"Loading model from {MODEL_PATH}", flush=True)

from vllm import LLM

llm = LLM(
    model=MODEL_PATH,
    max_model_len=MAX_MODEL_LEN,
    max_num_seqs=MAX_NUM_SEQS,
    max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    seed=SEED,
)
tokenizer = llm.get_tokenizer()
print("Tokenizer loaded.", flush=True)

# --- Method 1: convert_tokens_to_ids (works if they are in the vocab as-is) ---
think_start_id = tokenizer.convert_tokens_to_ids("<think>")
think_end_id = tokenizer.convert_tokens_to_ids("</think>")
print(f"\n[convert_tokens_to_ids]")
print(f"  <think>   -> {think_start_id}")
print(f"  </think>  -> {think_end_id}")

# --- Method 2: encode without special tokens ---
think_start_enc = tokenizer.encode("<think>", add_special_tokens=False)
think_end_enc = tokenizer.encode("</think>", add_special_tokens=False)
print(f"\n[encode, add_special_tokens=False]")
print(f"  <think>   -> {think_start_enc}")
print(f"  </think>  -> {think_end_enc}")

# --- Verification: decode the IDs back ---
print(f"\n[decode verification]")
if think_start_id is not None:
    print(f"  decode({think_start_id}) -> {repr(tokenizer.decode([think_start_id]))}")
if think_end_id is not None:
    print(f"  decode({think_end_id})   -> {repr(tokenizer.decode([think_end_id]))}")

# --- Determine single token IDs to hard-code ---
print("\n=== RESULTS TO HARD-CODE IN config.py ===")

# Prefer convert_tokens_to_ids if it gives a valid (non-UNK) single token
unk_id = tokenizer.unk_token_id
if think_start_id != unk_id and think_start_id is not None:
    print(f"THINK_START_TOKEN_ID = {think_start_id}")
elif len(think_start_enc) == 1:
    print(f"THINK_START_TOKEN_ID = {think_start_enc[0]}  # from encode()")
else:
    print(f"WARNING: <think> does not map to a single token: {think_start_enc}")

if think_end_id != unk_id and think_end_id is not None:
    print(f"THINK_END_TOKEN_ID   = {think_end_id}")
elif len(think_end_enc) == 1:
    print(f"THINK_END_TOKEN_ID   = {think_end_enc[0]}  # from encode()")
else:
    print(f"WARNING: </think> does not map to a single token: {think_end_enc}")

print("=========================================")
sys.stdout.flush()
