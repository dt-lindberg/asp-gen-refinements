"""vLLM engine for the Qwen3.6-35B-A3B GGUF smoke test.

Mirrors the root-level vllm_engine.py: Python-SDK LLM, explicit stop tokens,
<think>...</think> splitting. Adds language_model_only=True to skip the
vision tower on this multimodal checkpoint, with a fallback for older
vLLM builds that don't support the kwarg.
"""

import logging
import re
import time

from config import (
    SEED,
    THINKING,
    MAX_TOKENS,
    MAX_MODEL_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    TEMPERATURE,
    GPU_MEMORY_UTILIZATION,
    TOP_P,
    TOP_K,
    MIN_P,
    LANGUAGE_MODEL_ONLY,
)

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_thinking(text):
    # Paired <think>...</think> (GGUF tokenizers that ignore enable_thinking).
    if "<think>" in text:
        thinking = "\n".join(m.strip() for m in _THINK_RE.findall(text))
        response = _THINK_RE.sub("", text).strip()
        return thinking, response
    # Chat template already injected the opening <think>\n into the prompt,
    # so only </think> appears in the generated text (FP8 safetensors path).
    if "</think>" in text:
        thinking, _, response = text.partition("</think>")
        return thinking.strip(), response.strip()
    return "", text.strip()


class VLLMEngine:
    def __init__(self, model_path):
        from vllm import LLM, SamplingParams

        logger.info(f"Loading model from {model_path}")
        t0 = time.perf_counter()

        llm_kwargs = dict(
            model=model_path,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            seed=SEED,
        )
        if LANGUAGE_MODEL_ONLY:
            llm_kwargs["language_model_only"] = True

        try:
            self.llm = LLM(**llm_kwargs)
        except TypeError as e:
            if "language_model_only" not in str(e):
                raise
            logger.warning(
                "language_model_only kwarg unsupported; "
                "falling back to limit_mm_per_prompt={image:0, video:0}"
            )
            llm_kwargs.pop("language_model_only", None)
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
            self.llm = LLM(**llm_kwargs)

        logger.info(f"Model loaded in {time.perf_counter() - t0:.2f}s")

        self.tokenizer = self.llm.get_tokenizer()

        stop_token_ids: list[int] = []
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            stop_token_ids.append(eos_id)
        try:
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id is not None and im_end_id not in stop_token_ids:
                stop_token_ids.append(im_end_id)
                logger.info(f"Added <|im_end|> (id={im_end_id}) as stop token")
        except Exception:
            pass
        logger.info(f"Stop token ids: {stop_token_ids}")

        self.sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
            min_p=MIN_P,
            seed=SEED,
            stop_token_ids=stop_token_ids if stop_token_ids else None,
        )

    def _apply_template(self, messages):
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=THINKING,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_batch(self, messages_list):
        prompts = [self._apply_template(m) for m in messages_list]

        logger.info(f"Generating batch of {len(prompts)} prompts...")
        t0 = time.perf_counter()
        outputs = self.llm.generate(prompts, self.sampling_params)
        dt = time.perf_counter() - t0

        n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        logger.info(
            f"Generated {n_tokens} tokens in {dt:.2f}s ({n_tokens / dt:.2f} tok/s)"
        )

        return [_split_thinking(o.outputs[0].text) for o in outputs]
