#!/bin/bash
# Submit the test-split evaluation for the config in THIS worktree.
#
# The pipeline config (model, THINKING, temperature, max_tokens) lives in config.py,
# so which of the four configs you run is decided by the worktree you run this from
# -- not by a flag here. Each config has its own worktree and its own branch; run.job
# pins --chdir to the matching worktree. Run this once per worktree:
#
#   V1 no-think : asp-gen-refinements              (feature/self-hosted-llm)
#   V1 think    : asp-gen-refinements-thinking     (feature/qwen3-thinking-mode)
#   V2 no-think : asp-gen-refinements-V2           (asp-gen-refinements-V2)
#   V2 think    : asp-gen-refinements-V2-thinking  (feature/qwen3-thinking-mode-V2)
#
# Mirrors the train protocol: same 5 seeds as submit_runs.sh.
# The "100 eval puzzles" = test (50) + test_HA (50), submitted as separate jobs.
#
# Usage: bash submit_eval.sh [engine_label]
#   engine_label defaults to the branch's DEFAULT_ENGINE in config.py.

set -euo pipefail

SEEDS=(6610 38132 50 7529 13)
DATASETS=(test test_HA)

ENGINE="${1:-$(python3 -c 'import re,sys; \
src=open("config.py").read(); \
m=re.search(r"^DEFAULT_ENGINE\s*=\s*\"([^\"]+)\"", src, re.M); \
sys.stdout.write(m.group(1))')}"

BRANCH=$(git branch --show-current)
echo "Branch:  ${BRANCH}"
echo "Engine:  ${ENGINE}"
echo "Seeds:   ${SEEDS[*]}"
echo "Splits:  ${DATASETS[*]}"
echo

for DS in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOB_ID=$(sbatch --job-name="ASPEval_${DS}_${SEED}" \
            --export=ALL,RUN_SEED="${SEED}",RUN_DATASET="${DS}",RUN_NUM=-1 \
            run.job | awk '{print $NF}')
        echo "Submitted dataset=${DS} seed=${SEED}  job_id=${JOB_ID}  -> audit/vllm_${ENGINE}_${DS}_seed$(printf '%06d' "${SEED}")"
    done
done
