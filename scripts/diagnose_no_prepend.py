"""Experiment: let the model generate <think> itself instead of prepending it.

Identical to diagnose_thinking.py except apply_template does NOT append
<think>\\n to the prompt. The prompt ends with <|im_start|>assistant\\n,
and we observe whether the model generates token 151667 (<think>) as its
first output token.

Key questions:
  1. Does the model generate <think> (151667) as its first output token?
  2. Is the token stream now symmetric: 151667 at start, 151668 at end?
  3. Does _split_thinking() (Case 1) parse it correctly?
  4. Does the forced-cutoff path still work?
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


def apply_template_no_prepend(tokenizer, messages):
    """Apply chat template WITHOUT appending <think>.

    Prompt ends with <|im_start|>assistant\\n — the model must generate
    <think> itself if it wants to enter thinking mode.
    """
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return formatted


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
print(f"Model path : {MODEL_PATH}", flush=True)
print(f"Logit processor : diagnose_think_processor:DiagnoseThinkProcessor", flush=True)
print(f"Prompt style    : NO <think> prepend — model generates it freely", flush=True)

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

sep("Prompt inspection (no <think> prepend)")

test_messages = [{"role": "user", "content": "Write one ASP fact: alice is a person."}]
formatted_prompt = apply_template_no_prepend(tokenizer, test_messages)

print(f"\nFormatted prompt (repr, full):")
print(f"  {repr(formatted_prompt)}")
prompt_toks = tokenizer.encode(formatted_prompt, add_special_tokens=False)
print(f"\nPrompt token count : {len(prompt_toks)}")
print(f"Last 10 token IDs  : {prompt_toks[-10:]}")
last_10 = [tokenizer.decode([t], skip_special_tokens=False) for t in prompt_toks[-10:]]
print(f"Last 10 decoded    : {last_10}")

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

formatted_prompts = [apply_template_no_prepend(tokenizer, msgs) for msgs in prompts_raw]

print(f"\nRunning batch of {len(formatted_prompts)} prompts...", flush=True)
outputs = llm.generate(formatted_prompts, sampling_params)
print("Generation complete.", flush=True)

for output, label in zip(outputs, labels):
    log_output(tokenizer, output, label)

sep("DONE")
print("diagnose_no_prepend.py finished.", flush=True)
sys.stdout.flush()
