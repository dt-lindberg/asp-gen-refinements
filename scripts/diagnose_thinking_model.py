"""Diagnostic run with the Thinking-specific model variant.

Uses Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf — the model actually trained
for thinking tasks — instead of the Instruct variant.

No manual <think> insertion. We try the template as-is and let the model
handle the thinking boundary natively. The DiagnoseThinkProcessor is still
active so the budget is enforced and we can observe the forced-cutoff path.

Phases:
  1. Tokenizer probe  — inspect what the formatted prompt looks like and
                        whether token 151667 appears at the end naturally.
  2. Inference probe  — two prompts, full token-stream logging.
"""

import os
import re
import sys

DIAG_THINK_BUDGET = int(os.environ.get("DIAG_THINK_BUDGET", "30"))
DIAG_MAX_TOKENS   = int(os.environ.get("DIAG_MAX_TOKENS",   "120"))

print(f"DIAG_THINK_BUDGET = {DIAG_THINK_BUDGET}", flush=True)
print(f"DIAG_MAX_TOKENS   = {DIAG_MAX_TOKENS}",   flush=True)

# Resolve the cached model path at runtime.
from huggingface_hub import hf_hub_download
MODEL_PATH = hf_hub_download(
    repo_id="unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF",
    filename="Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf",
)
print(f"Model path : {MODEL_PATH}", flush=True)

from config import (
    MAX_MODEL_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    GPU_MEMORY_UTILIZATION,
    SEED,
    THINK_START_TOKEN_ID,
    THINK_END_TOKEN_ID,
)
from vllm import LLM, SamplingParams


# ── Helpers ──────────────────────────────────────────────────────────────────

def sep(title=""):
    bar = "=" * 70
    print(f"\n{bar}", flush=True)
    if title:
        print(f"  {title}", flush=True)
        print(bar, flush=True)


def apply_template(tokenizer, messages):
    """Apply chat template with enable_thinking=True, no manual insertion.

    For the Thinking model the template should handle the <think> marker
    natively. We fall back gracefully if enable_thinking is unsupported,
    but do NOT manually append anything.
    """
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        print("  [apply_template] enable_thinking not supported — using plain template", flush=True)
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return formatted


def apply_template_token_ids(tokenizer, messages):
    """Same as above but returns {'prompt_token_ids': [...]} for vLLM.

    Used as an alternative to see if tokenize=True gives a cleaner result.
    """
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    return {"prompt_token_ids": list(token_ids)}


def log_output(tokenizer, output, label):
    o = output.outputs[0]
    toks = list(o.token_ids)

    sep(f"Output: {label}")

    print(f"  Total tokens generated : {len(toks)}")
    print(f"  Finish reason          : {o.finish_reason}")

    has_start = THINK_START_TOKEN_ID in toks
    has_end   = THINK_END_TOKEN_ID   in toks
    print(f"  THINK_START in token_ids ({THINK_START_TOKEN_ID}): {has_start}", end="")
    if has_start:
        print(f"  at position {toks.index(THINK_START_TOKEN_ID)}", end="")
    print()
    print(f"  THINK_END   in token_ids ({THINK_END_TOKEN_ID}): {has_end}", end="")
    if has_end:
        print(f"  at position {toks.index(THINK_END_TOKEN_ID)}", end="")
    print()

    print(f"\n  repr(o.text):")
    print(f"    {repr(o.text)}")
    print(f"\n  '<think>'  in o.text : {'<think>'  in o.text}")
    print(f"  '</think>' in o.text : {'</think>' in o.text}")

    full_no_skip = tokenizer.decode(toks, skip_special_tokens=False)
    full_skip    = tokenizer.decode(toks, skip_special_tokens=True)
    print(f"\n  Full decode (skip_special_tokens=False):")
    print(f"    {repr(full_no_skip)}")
    print(f"\n  Full decode (skip_special_tokens=True):")
    print(f"    {repr(full_skip)}")

    print(f"\n  Per-token decode ({len(toks)} tokens):")
    for i, tok_id in enumerate(toks):
        no_skip = tokenizer.decode([tok_id], skip_special_tokens=False)
        skip    = tokenizer.decode([tok_id], skip_special_tokens=True)
        markers = ""
        if tok_id == THINK_END_TOKEN_ID:
            markers = "  <<<<  THINK_END"
        elif tok_id == THINK_START_TOKEN_ID:
            markers = "  <<<<  THINK_START"
        print(
            f"    [{i:4d}] id={tok_id:7d}  "
            f"no_skip={repr(no_skip):<25}  "
            f"skip={repr(skip)}{markers}"
        )

    # Replicate _split_thinking inline
    thinking_re = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    text = o.text
    match = thinking_re.search(text)
    if match:
        thinking = match.group(1).strip()
        response = thinking_re.sub("", text).strip()
        split_case = "Case 1 (full <think>...</think> in output)"
    else:
        end_idx = text.find("</think>")
        if end_idx != -1:
            thinking = text[:end_idx].strip()
            response = text[end_idx + len("</think>"):].strip()
            split_case = "Case 2 (<think> was in prompt; output starts with thinking)"
        else:
            thinking = ""
            response = text.strip()
            split_case = "Case 3 (no </think> found — THINKING LEAKED INTO RESPONSE)"

    print(f"\n  _split_thinking result:")
    print(f"    Case     : {split_case}")
    print(f"    thinking : {repr(thinking[:300])}{'...' if len(thinking) > 300 else ''}")
    print(f"    response : {repr(response[:300])}{'...' if len(response) > 300 else ''}")

    sys.stdout.flush()


# ── Load model ────────────────────────────────────────────────────────────────

sep("Loading model")
print(f"Logit processor : diagnose_think_processor:DiagnoseThinkProcessor", flush=True)

llm = LLM(
    model=MODEL_PATH,
    max_model_len=MAX_MODEL_LEN,
    max_num_seqs=MAX_NUM_SEQS,
    max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    seed=SEED,
    logits_processors=["diagnose_think_processor:DiagnoseThinkProcessor"],
)
tokenizer = llm.get_tokenizer()
print("Model and tokenizer loaded.", flush=True)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=DIAG_MAX_TOKENS,
    top_p=0.8,
    top_k=20,
    min_p=0.01,
)

# ── Phase 1: Tokenizer probe ──────────────────────────────────────────────────

sep("PHASE 1 — Tokenizer probe")

test_messages = [{"role": "user", "content": "Write one ASP fact: alice is a person."}]

# Probe A: text-based template
print("\n--- Approach A: apply_chat_template(tokenize=False, enable_thinking=True) ---")
formatted_text = apply_template(tokenizer, test_messages)
print(f"Formatted prompt (repr, full):")
print(f"  {repr(formatted_text)}")
prompt_toks_a = tokenizer.encode(formatted_text, add_special_tokens=False)
print(f"Token count : {len(prompt_toks_a)}")
print(f"Last 10 IDs : {prompt_toks_a[-10:]}")
print(f"Last 10 dec : {[tokenizer.decode([t], skip_special_tokens=False) for t in prompt_toks_a[-10:]]}")
print(f"Last token is THINK_START (151667): {prompt_toks_a[-1] == THINK_START_TOKEN_ID}")

# Probe B: token-id-based template
print("\n--- Approach B: apply_chat_template(tokenize=True, enable_thinking=True) ---")
prompt_dict_b = apply_template_token_ids(tokenizer, test_messages)
prompt_toks_b = prompt_dict_b["prompt_token_ids"]
print(f"Token count : {len(prompt_toks_b)}")
print(f"Last 10 IDs : {prompt_toks_b[-10:]}")
print(f"Last 10 dec : {[tokenizer.decode([t], skip_special_tokens=False) for t in prompt_toks_b[-10:]]}")
print(f"Last token is THINK_START (151667): {prompt_toks_b[-1] == THINK_START_TOKEN_ID}")

sys.stdout.flush()


# ── Phase 2: Inference — text-based prompt ────────────────────────────────────

sep(f"PHASE 2A — Inference: text template  (budget={DIAG_THINK_BUDGET}, max_tokens={DIAG_MAX_TOKENS})")

prompts_raw = [
    [{"role": "user", "content": "Write one ASP fact: alice is a person."}],
    [{"role": "user", "content": "Write one ASP constraint: no two people share the same seat."}],
]
labels = ["Prompt 0 — simple fact", "Prompt 1 — simple constraint"]

text_prompts = [apply_template(tokenizer, msgs) for msgs in prompts_raw]
print(f"\nRunning batch of {len(text_prompts)} prompts (text)...", flush=True)
outputs_a = llm.generate(text_prompts, sampling_params)
print("Done.", flush=True)

for output, label in zip(outputs_a, labels):
    log_output(tokenizer, output, f"{label}  [text template]")


# ── Phase 3: Inference — token-id-based prompt ───────────────────────────────

sep(f"PHASE 2B — Inference: token-id template  (budget={DIAG_THINK_BUDGET}, max_tokens={DIAG_MAX_TOKENS})")

tokenid_prompts = [apply_template_token_ids(tokenizer, msgs) for msgs in prompts_raw]
print(f"\nRunning batch of {len(tokenid_prompts)} prompts (token IDs)...", flush=True)
outputs_b = llm.generate(tokenid_prompts, sampling_params)
print("Done.", flush=True)

for output, label in zip(outputs_b, labels):
    log_output(tokenizer, output, f"{label}  [token-id template]")


sep("DONE")
print("diagnose_thinking_model.py finished.", flush=True)
sys.stdout.flush()
