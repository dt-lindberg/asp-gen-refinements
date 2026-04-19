# MODEL
# Qwen3.6-35B-A3B-FP8 is multimodal; LANGUAGE_MODEL_ONLY skips the vision tower.
REPO_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
LANGUAGE_MODEL_ONLY = True

# Seed for vLLM to allow deterministic cache reuse across runs
SEED = 132

# Enable/disable thinking mode
THINKING = True

# INFERENCE / SAMPLING
# MAX_TOKENS:               maximum number of output tokens per sequence
#   For thinking mode this budget covers both the <think>...</think> trace and
#   the final response.  Qwen3 thinking traces can be several thousand tokens,
#   so the limit is set high enough that the model terminates naturally.
# MAX_MODEL_LEN:            maximum input+output tokens a single sequence can occupy;
#   max_model_len * max_num_seqs = maximum tokens in batch (KV-cache limit)
# MAX_NUM_BATCHED_TOKENS:   maximum number of input tokens across a full batch
# MAX_NUM_SEQS:             maximum batch size (sequences processed in parallel)
# TOP_P:                    nucleus sampling — cumulative probability threshold
# TOP_K:                    only the top-K most likely tokens are considered per step
# MIN_P:                    minimum probability relative to top token to be considered
MAX_TOKENS = 81_920
MAX_MODEL_LEN = 94_000
MAX_NUM_BATCHED_TOKENS = 8192
MAX_NUM_SEQS = 25
GPU_MEMORY_UTILIZATION = 0.95
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 0.0
REPETITION_PENALTY = 1.0

# CLINGO
# CLINGO_MAX_MODELS: cap model enumeration to detect severely under-constrained programs without hanging
# CLINGO_TIMEOUT:    shared wall-clock budget for ground() and solve()
CLINGO_MAX_MODELS = 1001
CLINGO_TIMEOUT = 30.0

# REFINEMENT LOOP
# MAX_ATTEMPTS:                      number of refinement iterations
# SEVERELY_UNDERCONSTRAINED_THRESHOLD: programs with more answer sets than this get a simplified message
# MAX_VARIABLE_ATOMS:                limit variable atoms shown in semantic feedback
MAX_ATTEMPTS = 6
SEVERELY_UNDERCONSTRAINED_THRESHOLD = 1000
MAX_VARIABLE_ATOMS = 30

# PIPELINE
DEFAULT_ENGINE = "qwen36-35b-thinking"

PROMPT_PATHS = {
    "asp_fewshot": "prompts/0_gen_asp_fewshot.txt",
    "refinement_syntax": "prompts/7_refinement_syntax.txt",
    "refinement_semantic_unsat": "prompts/8_refinement_semantic_unsat.txt",
    "refinement_semantic_multi": "prompts/9_refinement_semantic_multi.txt",
}

# LOGGING
LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d > %(message)s"
)
ALLOWED_LOGGERS = ("__main__", "pipeline", "refinement_loop", "vllm_engine")
