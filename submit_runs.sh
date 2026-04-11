#!/bin/bash
# Submit one sbatch job per seed for refinement V1 comparison runs.
# Usage: bash submit_runs.sh

SEEDS=(006610 038132 000050 007529) # 000013

for SEED in "${SEEDS[@]}"; do
    JOB_ID=$(sbatch --job-name="ASPGen_${SEED}" --export=ALL,RUN_SEED="${SEED}" run.job | awk '{print $NF}')
    echo "Submitted seed=${SEED}  job_id=${JOB_ID}"
done
