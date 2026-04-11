import os
import threading

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
        self._vllm_engine = None

        for k, v in args.items():
            setattr(self, k, v)

    def _get_engine(self):
        if self._vllm_engine is None:
            from vllm_engine import VLLMEngine

            kwargs = dict(max_tokens=self.max_tokens, temperature=self.temperature)
            if hasattr(self, "seed") and self.seed is not None:
                kwargs["seed"] = self.seed
            self._vllm_engine = VLLMEngine(**kwargs)
        return self._vllm_engine

    def load_prompt(self):
        for kind in self.path_prompt:
            with open(self.path_prompt[kind], "r", encoding="utf-8") as f:
                self.prompt[kind] = f.read().strip()

    def gen_response_batch(self, kind, replaces):
        """Generate responses for a batch of puzzles.

        Returns:
            list of (prompt, thinking, response) tuples — one per puzzle.
            `prompt` is the final substituted user message string, `thinking`
            is extracted from <think>...</think> blocks (empty when THINKING
            is off), and `response` is the raw text with thinking stripped.
        """
        prompts = []
        for replace in replaces:
            prompt = self.prompt[kind]
            for k, v in replace.items():
                prompt = prompt.replace(k, v)
            prompts.append(prompt)

        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        generated = self._get_engine().generate_batch(messages_list)
        return [
            (prompts[i], thinking, response)
            for i, (thinking, response) in enumerate(generated)
        ]

    def gen_response_constraints_batch(self, kind, replaces):
        """Generate constraint responses for a batch of puzzles (multi-turn format).

        Returns:
            list of (prompt, thinking, response) tuples — one per puzzle.
            For this multi-turn step `prompt` is the full list of role/content
            message dicts that were sent to the LLM (including the in-context
            examples, which never carry thinking tokens).
        """
        messages_list = []

        for replace in replaces:
            prompt = self.prompt[kind]
            for k, v in replace.items():
                prompt = prompt.replace(k, v)

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

        generated = self._get_engine().generate_batch(messages_list)
        return [
            (messages_list[i], thinking, response)
            for i, (thinking, response) in enumerate(generated)
        ]

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
