#!/bin/bash
# Submit the test-split evaluation for the config on the CURRENTLY CHECKED-OUT branch.
#
# The pipeline config (model, THINKING, temperature, max_tokens) lives in config.py,
# so which of the four configs you run is decided by the branch you are on -- not by
# a flag here. Run this once per branch:
#
#   V1 no-think : feature/self-hosted-llm        (in asp-gen-refinements)
#   V1 think    : feature/qwen3-thinking-mode    (in asp-gen-refinements)
#   V2 no-think : asp-gen-refinements-V2         (in asp-gen-refinements-V2)
#   V2 think    : feature/qwen3-thinking-mode-V2 (in asp-gen-refinements-V2)
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
