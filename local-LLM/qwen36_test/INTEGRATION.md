# Integrating Qwen3.6-35B-A3B into an ASP-gen-style pipeline

A practical checklist for porting the existing Qwen3-30B-Thinking pipeline
(`asp-gen-refinements-V2` root) onto `Qwen/Qwen3.6-35B-A3B-FP8`. Validated on
Snellius H100 (gpu_h100 partition) on 2026-04-18.

---

## 1. Model choice

| Option | Fits 1×H100 80GB? | Quality | Status |
|---|---|---|---|
| `Qwen/Qwen3.6-35B-A3B` (BF16, ~72GB) | barely — no room for KV cache | highest | needs TP=2 on 2×H100 |
| **`Qwen/Qwen3.6-35B-A3B-FP8` (~36GB)** | **yes, comfortably** | **strictly better than current Q4_K_M baseline** | **recommended** |
| `unsloth/Qwen3.6-35B-A3B-GGUF` (various quants) | yes | varies | **do not use** — see pitfall #1 |

FP8 is the blessed path in the vLLM recipe
(<https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html>).

---

## 2. Environment

- **vLLM ≥ 0.19.0** (tested on 0.19.1). Older vLLM (e.g. 0.17.0 in the shared
  `asp-gen-refinements/.venv`) does not know the `qwen35moe` architecture.
- **transformers ≥ 5.5.x** — pulled in automatically by vLLM 0.19.1.
- **Python 3.13**, **CUDA 12.8** (Snellius modules: `2025`, `CUDA/12.8.0`,
  `Python/3.13.5-GCCcore-14.3.0`).
- **Use a fresh venv.** Do not upgrade the shared venv — the existing
  Qwen3-30B-Thinking pipeline works on vLLM 0.17.0 and upgrading has
  regression risk.

Install:
```bash
python3 -m venv <proj>/.venv
source <proj>/.venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir "vllm>=0.19.0"
pip install --no-cache-dir huggingface_hub
```

---

## 3. Weights download

- Use `snapshot_download` (52 files, includes sharded safetensors + config +
  tokenizer):
  ```python
  from huggingface_hub import snapshot_download
  path = snapshot_download(
      repo_id="Qwen/Qwen3.6-35B-A3B-FP8",
      allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.py"],
  )
  ```
- Pre-download on a CPU partition (`rome`, ~20s on Snellius) to keep the GPU
  slot busy only with inference.
- Cache path: `~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/<hash>/`.
  Mind home-quota — 36GB FP8 + any existing models can add up.

---

## 4. `vllm_engine.py` — what changes vs the root version

Copy the root `vllm_engine.py` as-is, then apply these three changes:

### 4.1 Pass `language_model_only=True` to `LLM(...)`

The checkpoint is multimodal; this flag skips the vision tower and frees
memory for KV cache. It's a kwarg on `LLM(...)` (flows through
`EngineArgs` → `MultiModalConfig.language_model_only`). Accepted on both
vLLM 0.17.0 and 0.19.1.

Keep a fallback for any intermediate vLLM versions that lack it:
```python
try:
    self.llm = LLM(**llm_kwargs)
except TypeError as e:
    if "language_model_only" not in str(e):
        raise
    llm_kwargs.pop("language_model_only", None)
    llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    self.llm = LLM(**llm_kwargs)
```

### 4.2 Fix `_split_thinking` — **this is the non-obvious one**

The FP8 tokenizer **honors `enable_thinking=True`** and injects `<think>\n`
into the prompt as part of the assistant-turn prefix. The model then only
emits the **closing** `</think>` plus the final answer. The root repo's
regex `<think>(.*?)</think>` requires both tags and silently returns empty
thinking.

Replace with a two-case handler:
```python
def _split_thinking(text):
    # Case A: paired tags (older GGUF tokenizers that ignore enable_thinking)
    if "<think>" in text:
        thinking = "\n".join(m.strip() for m in _THINK_RE.findall(text))
        response = _THINK_RE.sub("", text).strip()
        return thinking, response
    # Case B: chat template already injected opening <think>\n,
    # so only </think> appears in the generated text (FP8 path)
    if "</think>" in text:
        thinking, _, response = text.partition("</think>")
        return thinking.strip(), response.strip()
    return "", text.strip()
```

### 4.3 Stop tokens — keep as-is

Explicit stop-token handling from the root repo still applies:
- `tokenizer.eos_token_id`
- `tokenizer.convert_tokens_to_ids("<|im_end|>")`

On FP8 the single EOS id covers both (stop list `[248046]` was sufficient in
the smoke test), but keeping the `<|im_end|>` addition is harmless and
portable back to GGUF paths.

---

## 5. `config.py` values

Verified good on 1×H100 80GB, thinking mode on:

```python
REPO_ID = "Qwen/Qwen3.6-35B-A3B-FP8"

SEED = 132
THINKING = True

# Native context is 262K; 32K is comfortable for ASP prompts.
MAX_MODEL_LEN = 32768
MAX_TOKENS = 8192              # per-sequence output budget (thinking + answer)
MAX_NUM_BATCHED_TOKENS = 8192
MAX_NUM_SEQS = 4               # start small; increase if KV cache allows

# Qwen recipe: "thinking mode — coding" preset
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0

GPU_MEMORY_UTILIZATION = 0.92
LANGUAGE_MODEL_ONLY = True
```

The existing ASP pipeline's sampling constants already match this preset, so
no change there. For longer thinking traces on real puzzles, raise
`MAX_TOKENS` (root repo uses `81_920`).

---

## 6. SLURM job

Copy the root `run.job` header; the relevant bits:
```bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
module load 2025 CUDA/12.8.0 Python/3.13.5-GCCcore-14.3.0
source <proj>/.venv/bin/activate   # the NEW venv, not the shared one
```

---

## 7. Pitfalls (ordered by likelihood, most-first)

1. **GGUF path is broken.** `unsloth/Qwen3.6-35B-A3B-GGUF` tags its
   architecture as `qwen35moe`, which `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
   rejects (`ValueError: GGUF model with architecture qwen35moe is not
   supported yet.`) — fails during config extraction, before vLLM sees the
   weights. Use FP8 safetensors instead.
2. **Upgrading the shared venv breaks the 30B-Thinking pipeline.** Always
   build a parallel venv for 3.6 work. The two vLLM versions (0.17.0 vs
   0.19.1) coexist fine on disk in separate envs.
3. **Thinking appears empty.** See §4.2. If `thinking_chars=0` but the
   response contains `</think>` plain-text, it's the regex mismatch.
4. **KV cache / OOM.** If you increase `MAX_MODEL_LEN` or `MAX_NUM_SEQS` and
   see CUDA OOM, drop `GPU_MEMORY_UTILIZATION` to 0.90 first, then reduce
   `MAX_NUM_SEQS`. FP8 weights are ~36GB, so budget ~40GB for KV cache on an
   80GB H100.
5. **Home-dir quota.** 36GB FP8 snapshot + any other HF cache entries can
   overrun `$HOME`. Point `HF_HOME=/scratch/...` if needed.
6. **`language_model_only` silent no-op on very old vLLM.** The accept/reject
   behavior depends on build: pre-0.10 ignores the kwarg; 0.17.0+ honors it.
   The fallback in §4.1 covers both.

---

## 8. What the existing root repo gets to keep unchanged

- `pipeline.py` — engine-agnostic; just consumes `(prompt, thinking,
  response)` tuples.
- `refinement_loop.py`, `clingo_runner.py`, `main.py` — unchanged.
- Prompt files under `prompts/` — unchanged.
- Sampling params in `config.py` — unchanged.
- `run.job` template — only the venv `source` line needs to point at the new
  env.

Effectively the **only** code changes are `MODEL_PATH`/`REPO_ID`,
`_split_thinking`, and adding `language_model_only=True` to the `LLM(...)`
call.

---

## 9. Validated smoke-test outcome (2026-04-18)

- Install job (`rome`, 16 CPU): ~5 min incl. 36GB snapshot download (job 21982841).
- GPU smoke test (`gpu_h100`, 1 GPU): model load + 2 prompts, 871 generated
  tokens at **114.7 tok/s** (job 21982882).
- Thinking extraction worked after the §4.2 fix; both prompts produced
  coherent reasoning traces and correct final answers (`hello from qwen3.6`,
  `17 × 23 = 391`).

Reference files in this directory:
- `config.py`, `vllm_engine.py`, `test.py` — working reference implementation.
- `install_env.job`, `test_qwen36.job` — SLURM templates.
- `outputs/slurm_21982882.out`, `outputs/response_21982882.json` — validated run.
