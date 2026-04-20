"""Smoke test: load Qwen3.6-35B-A3B-FP8 via vLLM and run a few prompts."""

import json
import logging
import os
import sys
import time

from huggingface_hub import snapshot_download

from config import REPO_ID, FILENAME
from vllm_engine import VLLMEngine


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s > %(message)s",
)
logger = logging.getLogger("test")


PROMPTS = [
    [{"role": "user", "content": "Reply with exactly: hello from qwen3.6"}],
    [{"role": "user", "content": "What is 17 * 23? Think briefly, then answer."}],
]


def main():
    logger.info(f"Ensuring snapshot of {REPO_ID} is cached")
    t0 = time.perf_counter()
    model_path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.py"],
    )
    logger.info(f"Snapshot at: {model_path} (resolved in {time.perf_counter()-t0:.1f}s)")

    engine = VLLMEngine(model_path)

    outputs = engine.generate_batch(PROMPTS)

    results = []
    for msgs, (thinking, response) in zip(PROMPTS, outputs):
        user_msg = msgs[-1]["content"]
        logger.info(f"Prompt: {user_msg}")
        logger.info(f"Thinking chars: {len(thinking)}")
        logger.info(f"Response: {response!r}")
        results.append({
            "prompt": user_msg,
            "thinking_chars": len(thinking),
            "thinking_preview": thinking[:500],
            "response": response,
        })

    out_dir = "local-LLM/qwen36_test/outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"response_{os.getenv('SLURM_JOB_ID', 'local')}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Smoke test failed")
        sys.exit(1)
