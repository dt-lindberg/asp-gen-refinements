"""Experiment: prepend the actual <think> special token (151667) via prompt_token_ids.

Instead of appending the text string "<think>\\n" (which the GGUF tokenizer splits
into 3 subword tokens), we:
  1. Call apply_chat_template(tokenize=True) to get the prompt as a token ID list.
  2. Append THINK_START_TOKEN_ID (151667) directly to that list.
  3. Pass {"prompt_token_ids": [...]} to llm.generate() so vLLM never re-tokenises
     the prompt text and the special token arrives intact.

Expected improvement over the previous experiments:
  - Prompt ends with the real single token 151667, not ['<th','ink','>'].
  - Model receives the correct thinking-start signal.
  - Output token stream: [thinking content ...] 151668 [response ...]
  - _split_thinking() hits Case 2 (as before) but the KV-cache context is now
    consistent, which should reduce/eliminate the mid-sentence continuation bug.
"""

import os
import re
import sys

DIAG_THINK_BUDGET = int(os.environ.get("DIAG_THINK_BUDGET", "30"))
DIAG_MAX_TOKENS   = int(os.environ.get("DIAG_MAX_TOKENS",   "120"))

print(f"DIAG_THINK_BUDGET = {DIAG_THINK_BUDGET}", flush=True)
print(f"DIAG_MAX_TOKENS   = {DIAG_MAX_TOKENS}",   flush=True)

from config import (
    MODEL_PATH,
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


def apply_template_token_ids(tokenizer, messages):
    """Build prompt as token IDs and append THINK_START_TOKEN_ID (151667) directly.

    Returns a dict {"prompt_token_ids": [...]} suitable for llm.generate().
    The <think> token is injected as its actual vocabulary ID, not as text,
    so the GGUF tokenizer cannot split it into subwords.
    """
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    token_ids = list(token_ids) + [THINK_START_TOKEN_ID]
    return {"prompt_token_ids": token_ids}


def log_output(tokenizer, output, label):
    """Log complete token stream information for one RequestOutput."""
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

    # _split_thinking replicated inline
    thinking_re = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    text = o.text
    match = thinking_re.search(text)
    if match:
        thinking = match.group(1).strip()
        response = thinking_re.sub("", text).strip()
        split_case = "Case 1 (full <think>...</think> block in output)"
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
print(f"Model path      : {MODEL_PATH}", flush=True)
print(f"Logit processor : diagnose_think_processor:DiagnoseThinkProcessor", flush=True)
print(f"Prompt style    : tokenize=True + append THINK_START_TOKEN_ID ({THINK_START_TOKEN_ID})", flush=True)

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


# ── Prompt inspection ─────────────────────────────────────────────────────────

sep("Prompt inspection")

test_messages = [{"role": "user", "content": "Write one ASP fact: alice is a person."}]
prompt_dict = apply_template_token_ids(tokenizer, test_messages)
token_ids = prompt_dict["prompt_token_ids"]

print(f"\nPrompt token count : {len(token_ids)}")
print(f"Last 10 token IDs  : {token_ids[-10:]}")
last_10 = [tokenizer.decode([t], skip_special_tokens=False) for t in token_ids[-10:]]
print(f"Last 10 decoded    : {last_10}")
print(f"\nLast token is THINK_START_TOKEN_ID ({THINK_START_TOKEN_ID}): {token_ids[-1] == THINK_START_TOKEN_ID}")

sys.stdout.flush()


# ── Inference ─────────────────────────────────────────────────────────────────

sep(f"Inference  (DIAG_THINK_BUDGET={DIAG_THINK_BUDGET}, DIAG_MAX_TOKENS={DIAG_MAX_TOKENS})")

prompts_raw = [
    [{"role": "user", "content": "Write one ASP fact: alice is a person."}],
    [{"role": "user", "content": "Write one ASP constraint: no two people share the same seat."}],
]
labels = [
    "Prompt 0 — simple fact",
    "Prompt 1 — simple constraint",
]

formatted_prompts = [apply_template_token_ids(tokenizer, msgs) for msgs in prompts_raw]

print(f"\nRunning batch of {len(formatted_prompts)} prompts...", flush=True)
outputs = llm.generate(formatted_prompts, sampling_params)
print("Generation complete.", flush=True)

for output, label in zip(outputs, labels):
    log_output(tokenizer, output, label)

sep("DONE")
print("diagnose_token_prepend.py finished.", flush=True)
sys.stdout.flush()
