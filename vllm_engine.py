"""vLLM engine for batched local inference with Qwen3-30B-A3B."""

import os
import re
import time

from logger import setup_logging, get_logger

setup_logging(log_level=os.getenv("LOG_LEVEL", "debug"))
logger = get_logger(__name__)

MODEL_PATH = (
    "/home/dlindberg/.cache/huggingface/hub/"
    "models--unsloth--Qwen3-30B-A3B-Instruct-2507-GGUF/snapshots/"
    "eea7b2be5805a5f151f8847ede8e5f9a9284bf77/"
    "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class VLLMEngine:
    def __init__(self, max_tokens=1500, temperature=0.6, max_model_len=4400, max_num_seqs=64):
        from vllm import LLM, SamplingParams

        logger.info(f"Loading model from {MODEL_PATH}")
        t0 = time.perf_counter()
        self.llm = LLM(
            model=MODEL_PATH,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=0.93,
        )
        logger.info(f"Model loaded in {time.perf_counter() - t0:.2f}s")

        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _apply_template(self, messages):
        """Apply chat template, disabling thinking mode if supported."""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_batch(self, messages_list):
        """Generate responses for a batch of conversations.

        Args:
            messages_list: list of conversations, each a list of role/content dicts.

        Returns:
            list of response strings (thinking tokens stripped).
        """
        formatted = [self._apply_template(msgs) for msgs in messages_list]

        logger.info(f"Generating batch of {len(formatted)} prompts...")
        t0 = time.perf_counter()
        outputs = self.llm.generate(formatted, self.sampling_params)
        t_gen = time.perf_counter() - t0

        n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        logger.info(
            f"Generated {n_tokens} tokens in {t_gen:.2f}s ({n_tokens / t_gen:.2f} tok/s)"
        )

        return [_THINK_RE.sub("", o.outputs[0].text).strip() for o in outputs]
