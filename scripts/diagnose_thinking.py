"""Diagnose the exact token stream for thinking mode.

Investigates whether <think>/<think> special tokens survive into
o.outputs[0].text, or are stripped by skip_special_tokens, and what
the raw token stream looks like at the thinking/response boundary.

Controlled by env vars (set in diagnose_thinking.job):
  DIAG_THINK_BUDGET  — thinking budget passed to DiagnoseThinkProcessor
  DIAG_MAX_TOKENS    — max_tokens for SamplingParams

Phases:
  1. Tokenizer probe   — no inference; checks special-token IDs and decode behaviour
  2. Inference probe   — runs two prompts; logs full token stream, .text, and
                         per-token decodes with/without skip_special_tokens
"""

import os
import sys

DIAG_THINK_BUDGET = int(os.environ.get("DIAG_THINK_BUDGET", "30"))
DIAG_MAX_TOKENS = int(os.environ.get("DIAG_MAX_TOKENS", "120"))

print(f"DIAG_THINK_BUDGET = {DIAG_THINK_BUDGET}", flush=True)
print(f"DIAG_MAX_TOKENS   = {DIAG_MAX_TOKENS}", flush=True)

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


def apply_template(tokenizer, messages):
    """Apply chat template and manually append <think> for thinking mode."""
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if not formatted.rstrip("\n").endswith("<think>"):
        formatted = formatted.rstrip("\n") + "<think>\n"
    return formatted


def log_output(tokenizer, output, label):
    """Log complete token stream information for one RequestOutput."""
    o = output.outputs[0]
    toks = list(o.token_ids)

    sep(f"Output: {label}")

    # Basic counts
    print(f"  Total tokens generated : {len(toks)}")
    print(f"  Finish reason          : {o.finish_reason}")

    # Presence of think tokens
    has_start = THINK_START_TOKEN_ID in toks
    has_end = THINK_END_TOKEN_ID in toks
    print(f"  THINK_START in token_ids ({THINK_START_TOKEN_ID}): {has_start}", end="")
    if has_start:
        print(f"  at position {toks.index(THINK_START_TOKEN_ID)}", end="")
    print()
    print(f"  THINK_END   in token_ids ({THINK_END_TOKEN_ID}): {has_end}", end="")
    if has_end:
        print(f"  at position {toks.index(THINK_END_TOKEN_ID)}", end="")
    print()

    # Exact .text content
    print(f"\n  repr(o.text):")
    print(f"    {repr(o.text)}")
    print(f"\n  '<think>'  in o.text : {'<think>'  in o.text}")
    print(f"  '</think>' in o.text : {'</think>' in o.text}")

    # Full decode: no skip vs skip
    full_no_skip = tokenizer.decode(toks, skip_special_tokens=False)
    full_skip    = tokenizer.decode(toks, skip_special_tokens=True)
    print(f"\n  Full decode (skip_special_tokens=False):")
    print(f"    {repr(full_no_skip)}")
    print(f"\n  Full decode (skip_special_tokens=True):")
    print(f"    {repr(full_skip)}")

    print(f"\n  '<think>'  in full_no_skip : {'<think>'  in full_no_skip}")
    print(f"  '</think>' in full_no_skip : {'</think>' in full_no_skip}")
    print(f"  '<think>'  in full_skip    : {'<think>'  in full_skip}")
    print(f"  '</think>' in full_skip    : {'</think>' in full_skip}")

    # Per-token decode (all tokens — short outputs expected)
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

    # _split_thinking result (replicated inline — no import needed)
    import re
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
print(f"Using logit processor: diagnose_think_processor:DiagnoseThinkProcessor", flush=True)

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

sep("PHASE 1 — Tokenizer probe (no inference)")

print(f"\nConfig token IDs (from config.py):")
print(f"  THINK_START_TOKEN_ID = {THINK_START_TOKEN_ID}")
print(f"  THINK_END_TOKEN_ID   = {THINK_END_TOKEN_ID}")

# Check whether these IDs are registered as special tokens
all_special = set(getattr(tokenizer, "all_special_ids", []))
print(f"\nall_special_ids contains THINK_START ({THINK_START_TOKEN_ID}): {THINK_START_TOKEN_ID in all_special}")
print(f"all_special_ids contains THINK_END   ({THINK_END_TOKEN_ID}):   {THINK_END_TOKEN_ID   in all_special}")

# Decode each think token with and without skip_special_tokens
print()
for skip in (False, True):
    s_start = tokenizer.decode([THINK_START_TOKEN_ID], skip_special_tokens=skip)
    s_end   = tokenizer.decode([THINK_END_TOKEN_ID],   skip_special_tokens=skip)
    print(f"decode(skip_special_tokens={skip}):")
    print(f"  THINK_START -> {repr(s_start)}")
    print(f"  THINK_END   -> {repr(s_end)}")

# Encode round-trip for the exact strings used in _split_thinking and _apply_template
print()
for text in ("<think>", "</think>", "<think>\n"):
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"encode({repr(text):<16}, add_special_tokens=False) -> {ids}")

print()
for text in ("<think>", "</think>"):
    tok_id = tokenizer.convert_tokens_to_ids(text)
    print(f"convert_tokens_to_ids({repr(text):<12}) -> {tok_id}")

# Show the formatted prompt so we can see exactly what is sent to the model
test_messages = [{"role": "user", "content": "Write one ASP fact: alice is a person."}]
formatted_prompt = apply_template(tokenizer, test_messages)
print(f"\nFormatted prompt for test messages:")
print(f"  (repr of last 300 chars): {repr(formatted_prompt[-300:])}")
prompt_toks = tokenizer.encode(formatted_prompt, add_special_tokens=False)
print(f"  Prompt token count: {len(prompt_toks)}")
print(f"  Last 10 prompt token IDs: {prompt_toks[-10:]}")
last_10_decoded = [tokenizer.decode([t], skip_special_tokens=False) for t in prompt_toks[-10:]]
print(f"  Last 10 decoded (no_skip): {last_10_decoded}")

sys.stdout.flush()


# ── Phase 2: Inference probe ──────────────────────────────────────────────────

sep(f"PHASE 2 — Inference probe  (DIAG_THINK_BUDGET={DIAG_THINK_BUDGET}, DIAG_MAX_TOKENS={DIAG_MAX_TOKENS})")

# Two prompts: one very short task, one slightly more involved
prompts_raw = [
    [{"role": "user", "content": "Write one ASP fact: alice is a person."}],
    [{"role": "user", "content": "Write one ASP constraint: no two people share the same seat."}],
]
labels = [
    "Prompt 0 — simple fact",
    "Prompt 1 — simple constraint",
]

formatted_prompts = [apply_template(tokenizer, msgs) for msgs in prompts_raw]

print(f"\nRunning batch of {len(formatted_prompts)} prompts...", flush=True)
outputs = llm.generate(formatted_prompts, sampling_params)
print("Generation complete.", flush=True)

for i, (output, label) in enumerate(zip(outputs, labels)):
    log_output(tokenizer, output, label)

sep("DONE")
print("diagnose_thinking.py finished.", flush=True)
sys.stdout.flush()
