"""Config for the Qwen3.6-35B-A3B GGUF smoke test."""

REPO_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
FILENAME = None  # use full repo snapshot (safetensors); None means vLLM resolves

SEED = 132
THINKING = True

MAX_TOKENS = 8192
MAX_MODEL_LEN = 32768
MAX_NUM_BATCHED_TOKENS = 8192
MAX_NUM_SEQS = 4
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
GPU_MEMORY_UTILIZATION = 0.92

LANGUAGE_MODEL_ONLY = True
