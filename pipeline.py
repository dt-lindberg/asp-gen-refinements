import json
import os
import threading
import pandas as pd

from time import strftime
from clingo.control import Control
from clingo.symbol import parse_term

from logger import setup_logging, get_logger
from config import (
    DEFAULT_ENGINE,
    TEMPERATURE,
    MAX_TOKENS,
    CLINGO_MAX_MODELS,
    CLINGO_TIMEOUT,
    CONSTRAINTS_SYSTEM,
    CONSTRAINTS_ASSISTANT_ACK,
    THINKING_OVERFLOW,
)

setup_logging(log_level=os.getenv("LOG_LEVEL", "debug"))
logger = get_logger(__name__)


# clingo context used to define python functions in clingo
class Context:
    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(" "):
            ret.append(parse_term(term))
        return ret


class Pipeline:
    def __init__(self, args):
        self.engine = DEFAULT_ENGINE
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.path_prompt = {}
        self.prompt = {}
        self.path_cache = {}
        self.cache = {}
        self._vllm_engine = None
        os.makedirs("mistakes", exist_ok=True)
        self.path_mistakes = f"mistakes/mistakes_{strftime('%m%d_%H%M%S')}.xlsx"
        self.mistakes = []

        for k, v in args.items():
            setattr(self, k, v)

    def _get_engine(self):
        if self._vllm_engine is None:
            from vllm_engine import VLLMEngine

            self._vllm_engine = VLLMEngine(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        return self._vllm_engine

    def load_prompt(self):
        for kind in self.path_prompt:
            with open(self.path_prompt[kind], "r", encoding="utf-8") as f:
                self.prompt[kind] = f.read().strip()

    def load_cache(self):
        for kind in self.path_cache:
            if os.path.isfile(self.path_cache[kind]):
                with open(self.path_cache[kind], "r") as f:
                    self.cache[kind] = json.load(f)
            else:
                self.cache[kind] = {}

    def save_cache(self):
        os.makedirs("caches", exist_ok=True)
        for kind in self.path_cache:
            with open(self.path_cache[kind], "w") as f:
                json.dump(self.cache[kind], f)

    @staticmethod
    def _cache_response(entry):
        """Extract response text from a cache entry (supports old string format and new dict format)."""
        return entry["response"] if isinstance(entry, dict) else entry

    def gen_response_batch(self, kind, replaces):
        """Generate responses for a batch of puzzles.

        Args:
            kind: prompt kind string.
            replaces: list of replace dicts (one per puzzle).

        Returns:
            list of response text strings.
        """
        prompts = []
        for replace in replaces:
            prompt = self.prompt[kind]
            for k, v in replace.items():
                prompt = prompt.replace(k, v)
            prompts.append(prompt)

        responses = [None] * len(prompts)
        miss_indices = []
        miss_messages = []

        for i, prompt in enumerate(prompts):
            if prompt in self.cache[kind]:
                responses[i] = self._cache_response(self.cache[kind][prompt])
            else:
                miss_indices.append(i)
                miss_messages.append([{"role": "user", "content": prompt}])

        if miss_messages:
            generated = self._get_engine().generate_batch(miss_messages)
            for idx, (thinking, resp) in zip(miss_indices, generated):
                if resp == THINKING_OVERFLOW:
                    logger.warning(
                        f"Thinking overflow on prompt {idx} (kind={kind}): "
                        "model exhausted token budget before closing </think>"
                    )
                self.cache[kind][prompts[idx]] = {"response": resp, "thinking": thinking}
                responses[idx] = resp
            self.save_cache()

        return responses

    def gen_response_constraints_batch(self, kind, replaces):
        """Generate constraint responses for a batch of puzzles (multi-turn format).

        Args:
            kind: prompt kind string.
            replaces: list of replace dicts (one per puzzle).

        Returns:
            list of response text strings.
        """
        prompts = []
        messages_list = []

        for replace in replaces:
            prompt = self.prompt[kind]
            for k, v in replace.items():
                prompt = prompt.replace(k, v)
            prompts.append(prompt)

            general, ex1, ex2, ex3 = prompt.split("\n\nProblem ")
            ex1, response1 = ex1.split("\n\nConstraints:\n")
            ex2, response2 = ex2.split("\n\nConstraints:\n")
            ex1 = "Problem " + ex1 + "\n\nConstraints:"
            ex2 = "Problem " + ex2 + "\n\nConstraints:"
            ex3 = "Problem " + ex3
            messages = [
                {"role": "system", "content": CONSTRAINTS_SYSTEM},
                {"role": "user", "content": general},
                {"role": "assistant", "content": CONSTRAINTS_ASSISTANT_ACK},
                {"role": "user", "content": ex1},
                {"role": "assistant", "content": response1},
                {"role": "user", "content": ex2},
                {"role": "assistant", "content": response2},
                {"role": "user", "content": ex3},
            ]
            messages_list.append(messages)

        responses = [None] * len(prompts)
        miss_indices = []
        miss_messages = []

        for i, prompt in enumerate(prompts):
            if prompt in self.cache[kind]:
                responses[i] = self._cache_response(self.cache[kind][prompt])
            else:
                miss_indices.append(i)
                miss_messages.append(messages_list[i])

        if miss_messages:
            generated = self._get_engine().generate_batch(miss_messages)
            for idx, (thinking, resp) in zip(miss_indices, generated):
                if resp == THINKING_OVERFLOW:
                    logger.warning(
                        f"Thinking overflow on prompt {idx} (kind={kind}): "
                        "model exhausted token budget before closing </think>"
                    )
                self.cache[kind][prompts[idx]] = {"response": resp, "thinking": thinking}
                responses[idx] = resp
            self.save_cache()

        return responses

    def gen_response_raw_batch(self, kind, prompts):
        """Generate responses for pre-built prompt strings."""
        responses = [None] * len(prompts)
        miss_indices = []
        miss_messages = []

        for i, prompt in enumerate(prompts):
            if prompt in self.cache[kind]:
                responses[i] = self._cache_response(self.cache[kind][prompt])
            else:
                miss_indices.append(i)
                miss_messages.append([{"role": "user", "content": prompt}])

        if miss_messages:
            generated = self._get_engine().generate_batch(miss_messages)
            for idx, (thinking, resp) in zip(miss_indices, generated):
                if resp == THINKING_OVERFLOW:
                    logger.warning(
                        f"Thinking overflow on prompt {idx} (kind={kind}): "
                        "model exhausted token budget before closing </think>"
                    )
                self.cache[kind][prompts[idx]] = {"response": resp, "thinking": thinking}
                responses[idx] = resp
            self.save_cache()

        return responses

    def gen_response(self, kind, replace):
        """Single-puzzle convenience wrapper around gen_response_batch."""
        return self.gen_response_batch(kind, [replace])[0]

    def gen_response_constraints(self, kind, replace):
        """Single-puzzle convenience wrapper around gen_response_constraints_batch."""
        return self.gen_response_constraints_batch(kind, [replace])[0]

    def gen_answer_set(self, program, opt=False):
        """Run Clingo to find answer sets.

        Returns:
            (None, list_of_answer_sets) on success, (RuntimeError, messages) on parse error.
        """
        clingo_messages = []

        def _clingo_logger(code, message):
            clingo_messages.append((code, message))

        clingo_control = Control(
            [str(CLINGO_MAX_MODELS), "--warn=none", "--opt-mode=optN", "-t", "4"],
            logger=_clingo_logger,
        )
        models = []
        try:
            program_clean = program.encode("ascii", errors="replace").decode("ascii")
            logger.debug(f"Cleaned program passed to Clingo:\n{program_clean}")
            clingo_control.add("base", [], program_clean)
        except RuntimeError as e:
            logger.info(
                f"Clingo parsing failed with error={e} and "
                f"#messages={len(clingo_messages)}"
            )
            logger.debug(f"Messages:\n{clingo_messages}")
            return RuntimeError, clingo_messages
        except Exception as e:
            logger.error(f"Clingo failed with error={e}")
            return RuntimeError, []

        # ground() can loop forever on LLM-generated programs that use recursive
        # arithmetic rules deriving out-of-domain atoms (e.g. month = month - 2
        # recursively). Run it in a daemon thread so we can time out and move on.
        logger.debug("Clingo: starting ground()")
        ground_exc = [None]
        ground_done = threading.Event()

        def _do_ground():
            try:
                clingo_control.ground([("base", [])], context=Context())
            except Exception as e:
                ground_exc[0] = e
            finally:
                ground_done.set()

        threading.Thread(target=_do_ground, daemon=True).start()

        if not ground_done.wait(CLINGO_TIMEOUT):
            logger.warning(f"Clingo ground() timed out after {CLINGO_TIMEOUT}s, skipping puzzle")
            return RuntimeError, []

        if ground_exc[0] is not None:
            e = ground_exc[0]
            if isinstance(e, RuntimeError):
                logger.info(
                    f"Clingo grounding failed with error={e} and "
                    f"#messages={len(clingo_messages)}"
                )
                logger.debug(f"Messages:\n{clingo_messages}")
                return RuntimeError, clingo_messages
            logger.error(f"Clingo failed with error={e}")
            return RuntimeError, []

        logger.debug("Clingo: ground() complete, starting solve()")

        if opt:
            on_model_cb = lambda model: (
                models.append(model.symbols(atoms=True))
                if model.optimality_proven
                else None
            )
        else:
            on_model_cb = lambda model: models.append(model.symbols(atoms=True))

        with clingo_control.solve(on_model=on_model_cb, async_=True) as handle:
            finished = handle.wait(CLINGO_TIMEOUT)
            if not finished:
                handle.cancel()
                handle.wait()
                logger.warning(
                    f"Clingo solve timed out after {CLINGO_TIMEOUT}s "
                    f"({len(models)} models found so far), cancelling"
                )

        models = [[str(atom) for atom in model] for model in models]
        return None, models

    def get_reasoning(self, kind, replace):
        """Returns empty string — reasoning traces not available for local vLLM backend."""
        return ""

    def save_mistakes(self, mistake_cols):
        df = pd.DataFrame(self.mistakes, columns=mistake_cols)
        writer = pd.ExcelWriter(self.path_mistakes)
        df.to_excel(writer, sheet_name="results")
        for col_idx in range(2, 10):
            writer.sheets["results"].set_column(col_idx, col_idx, 40)
        writer.close()
