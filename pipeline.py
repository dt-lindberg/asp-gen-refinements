import json
import os
import pandas as pd

import openai
from dotenv import load_dotenv
from time import strftime
from clingo.control import Control
from clingo.symbol import parse_term
from google import genai
from google.genai import types
from groq import Groq

from logger import setup_logging, get_logger

load_dotenv()

setup_logging(log_level=os.getenv("LOG_LEVEL", "debug"))
logger = get_logger(__name__)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# clingo context used to define python functions in clingo
class Context:
    # get features/words from a string of space separated words
    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(" "):
            ret.append(parse_term(term))
        return ret


class Pipeline:
    def __init__(self, args):
        self.asp_program = ""
        ###########
        # Gemini API
        ###########
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.thinking_level = "medium"
        ###########
        # Groq API
        ###########
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        ###########
        # GPT-3
        ###########
        self.engine = "text-davinci-003"
        self.temperature = 0.0
        self.max_tokens = 1500
        self.path_prompt = {}  # store the mapping from kind (str) to path of prompt file (str)
        self.prompt = {}  # a mapping from prompt kind (str) to the prompt (str)
        ###########
        # Cache
        ###########
        self.path_cache = {}  # store the mapping from kind (str) to path of cache file (str)
        self.cache = {}  # store the mapping from kind (str) to cached responses (dictionary)
        os.makedirs("mistakes", exist_ok=True)
        self.path_mistakes = f"mistakes/mistakes_{strftime('%m%d_%H%M%S')}.xlsx"  # file to store the wrong pridictions
        self.mistakes = []  # store the wrong predictions

        for k, v in args.items():
            setattr(self, k, v)
        # initialze openai account
        openai.api_key = os.getenv("OPENAI_API_KEY")

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
        for kind in self.path_cache:
            with open(self.path_cache[kind], "w") as f:
                json.dump(self.cache[kind], f)

    # take a kind and replace (dictionary), return the GPT3 response
    def gen_response(self, kind, replace):
        # obtain the whole prompt
        prompt = self.prompt[kind]
        for k in replace:
            prompt = prompt.replace(k, replace[k])
        # generate and cache the response in cache if it's not cached before
        if prompt not in self.cache[kind]:
            try:
                if self.engine == "gemini-3-flash-preview":
                    # Define contents suitable for Gemini API, equivalent to messages
                    contents = [
                        types.Content(
                            role="user",
                            parts=[types.Part(text=prompt)],
                        )
                    ]
                    try:
                        gemini_response = self.client.models.generate_content(
                            model=self.engine,
                            contents=contents,
                            config=types.GenerateContentConfig(
                                thinking_config=types.ThinkingConfig(
                                    thinking_level=self.thinking_level
                                ),
                                temperature=self.temperature,
                                max_output_tokens=self.max_tokens,
                            ),
                        )
                        self.cache[kind][prompt] = json.loads(
                            gemini_response.model_dump_json()
                        )
                    except Exception as e:
                        logger.error(f"Gemini API failed with error={e}")
                elif self.engine == "gpt-oss-120b":
                    messages = [{"role": "user", "content": prompt}]
                    try:
                        response = self.groq_client.chat.completions.create(
                            messages=messages,
                            model="openai/gpt-oss-120b",
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                        self.cache[kind][prompt] = response.model_dump()
                    except Exception as e:
                        logger.error(f"Groq API failed with error={e}")
                elif self.engine == "gpt-4":
                    messages = [{"role": "user", "content": prompt}]
                    try:
                        self.cache[kind][prompt] = openai.ChatCompletion.create(
                            messages=messages,
                            model="gpt-4",
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                    except Exception as e:
                        logger.error(f"GPT API failed with error={e}")
                else:
                    self.cache[kind][prompt] = openai.Completion.create(
                        prompt=prompt,
                        engine=self.engine,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )

                # Save response to cache
                self.save_cache()

            except Exception as e:
                logger.error(f"Calling APIs failed with error={e}")
                breakpoint()
                self.cache[kind][prompt] = None
        if self.engine == "gemini-3-flash-preview":
            return self.cache[kind][prompt]["candidates"][0]["content"]["parts"][0][
                "text"
            ].strip()
        elif self.engine in ("gpt-4", "gpt-oss-120b"):
            return self.cache[kind][prompt]["choices"][0]["message"]["content"].strip()
        return self.cache[kind][prompt]["choices"][0]["text"].strip()

    # take a kind and replace (dictionary), return the GPT3 response
    def gen_response_constraints(self, kind, replace):
        # obtain the whole prompt
        prompt = self.prompt[kind]
        for k in replace:
            prompt = prompt.replace(k, replace[k])
        # generate and cache the response in cache if it's not cached before
        if prompt not in self.cache[kind]:
            try:
                if self.engine == "gemini-3-flash-preview":
                    general, ex1, ex2, ex3 = prompt.split("\n\nProblem ")
                    ex1, response1 = ex1.split("\n\nConstraints in UTF-8 encoding:\n")
                    ex2, response2 = ex2.split("\n\nConstraints in UTF-8 encoding:\n")
                    ex1 = "Problem " + ex1 + "\n\nConstraints in UTF-8 encoding:"
                    ex2 = "Problem " + ex2 + "\n\nConstraints in UTF-8 encoding:"
                    ex3 = "Problem " + ex3

                    system_instruction = "You are a semantic parser to turn clues in a problem into logical rules using only provided constants and predicates."

                    contents = [
                        types.Content(role="user", parts=[types.Part(text=general)]),
                        types.Content(
                            role="model",
                            parts=[
                                types.Part(
                                    text="Ok. I will only write constraints under the provided forms."
                                )
                            ],
                        ),
                        types.Content(role="user", parts=[types.Part(text=ex1)]),
                        types.Content(role="model", parts=[types.Part(text=response1)]),
                        types.Content(role="user", parts=[types.Part(text=ex2)]),
                        types.Content(role="model", parts=[types.Part(text=response2)]),
                        types.Content(role="user", parts=[types.Part(text=ex3)]),
                    ]

                    gemini_response = self.client.models.generate_content(
                        model=self.engine,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            thinking_config=types.ThinkingConfig(
                                thinking_level=self.thinking_level
                            ),
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                        ),
                    )

                    self.cache[kind][prompt] = json.loads(
                        gemini_response.model_dump_json()
                    )

                elif self.engine == "gpt-oss-120b":
                    general, ex1, ex2, ex3 = prompt.split("\n\nProblem ")
                    ex1, response1 = ex1.split("\n\nConstraints in UTF-8 encoding:\n")
                    ex2, response2 = ex2.split("\n\nConstraints in UTF-8 encoding:\n")
                    ex1 = "Problem " + ex1 + "\n\nConstraints in UTF-8 encoding:"
                    ex2 = "Problem " + ex2 + "\n\nConstraints in UTF-8 encoding:"
                    ex3 = "Problem " + ex3
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a semantic parser to turn clues in a problem into logical rules using only provided constants and predicates.",
                        },
                        {"role": "user", "content": general},
                        {
                            "role": "assistant",
                            "content": "Ok. I will only write constraints under the provided forms.",
                        },
                        {"role": "user", "content": ex1},
                        {"role": "assistant", "content": response1},
                        {"role": "user", "content": ex2},
                        {"role": "assistant", "content": response2},
                        {"role": "user", "content": ex3},
                    ]
                    response = self.groq_client.chat.completions.create(
                        messages=messages,
                        model="openai/gpt-oss-120b",
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    self.cache[kind][prompt] = response.model_dump()
                elif self.engine == "gpt-4":
                    # split prompt into different messages
                    general, ex1, ex2, ex3 = prompt.split("\n\nProblem ")
                    ex1, response1 = ex1.split("\n\nConstraints in UTF-8 encoding:\n")
                    ex2, response2 = ex2.split("\n\nConstraints in UTF-8 encoding:\n")
                    ex1 = "Problem " + ex1 + "\n\nConstraints in UTF-8 encoding:"
                    ex2 = "Problem " + ex2 + "\n\nConstraints in UTF-8 encoding:"
                    ex3 = "Problem " + ex3
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a semantic parser to turn clues in a problem into logical rules using only provided constants and predicates.",
                        },
                        {"role": "system", "name": "example_user", "content": general},
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": "Ok. I will only write constraints under the provided forms.",
                        },
                        {"role": "system", "name": "example_user", "content": ex1},
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": response1,
                        },
                        {"role": "system", "name": "example_user", "content": ex2},
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": response2,
                        },
                        {"role": "user", "content": ex3},
                    ]
                    self.cache[kind][prompt] = openai.ChatCompletion.create(
                        messages=messages,
                        model="gpt-4",
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                else:
                    self.cache[kind][prompt] = openai.Completion.create(
                        prompt=prompt,
                        engine=self.engine,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                self.save_cache()
            except Exception as e:
                logger.error(f"Calling APIs failed with error={e}")
                breakpoint()
                self.cache[kind][prompt] = None
        if self.engine == "gemini-3-flash-preview":
            return self.cache[kind][prompt]["candidates"][0]["content"]["parts"][0][
                "text"
            ].strip()
        elif self.engine in ("gpt-4", "gpt-oss-120b"):
            return self.cache[kind][prompt]["choices"][0]["message"]["content"].strip()
        return self.cache[kind][prompt]["choices"][0]["text"].strip()

    # take a kind and replace (dictionary), return the GPT response
    # NOTE: never called, no Gemini implementation made
    def gen_response_bk(self, kind, replace):
        # obtain the whole prompt
        prompt = self.prompt[kind]
        for k in replace:
            prompt = prompt.replace(k, replace[k])
        # generate and cache the response in cache if it's not cached before
        if prompt not in self.cache[kind]:
            try:
                self.cache[kind][prompt] = openai.Completion.create(
                    prompt=prompt,
                    engine=self.engine,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                self.save_cache()
            except Exception as e:
                print(e)
                breakpoint()
                self.cache[kind][prompt] = None
        return self.cache[kind][prompt]["choices"][0]["text"].strip()

    # use ASP (clingo) to find answer sets
    def gen_answer_set(self, program, opt=False):
        """
        Returns a tuple of (error, answer_sets), error=None if no error.

        Args:
            program (str): a string of ASP program
            opt (bool): if true, only optimal answer sets are returned
                        leave it to False when there is no weak constraint
        """
        clingo_messages = []

        def _clingo_logger(code, message):
            """Clingo logger to catch and format syntax errors"""
            clingo_messages.append((code, message))

        clingo_control = Control(
            ["0", "--warn=none", "--opt-mode=optN", "-t", "4"], logger=_clingo_logger
        )
        models = []
        # breakpoint()
        try:
            # Encode the program as ASCII before passing it to Clingo
            program_clean = program.encode("ascii", errors="replace").decode("ascii")

            logger.debug(f"Cleaned program passed to Clingo:\n{program_clean}")

            clingo_control.add("base", [], program_clean)
            clingo_control.ground([("base", [])], context=Context())
        # Catch syntax errors
        except RuntimeError as e:
            logger.info(
                f"Clingo parsing/grounding failed with error={e} and #messages={len(clingo_messages)}"
            )
            logger.debug(f"Messages:\n{clingo_messages}")
            return RuntimeError, clingo_messages
        except Exception as e:
            logger.error(f"Clingo failed with error={e}")
            return RuntimeError, []

        if opt:
            clingo_control.solve(
                on_model=lambda model: (
                    models.append(model.symbols(atoms=True))
                    if model.optimality_proven
                    else None
                )
            )
        else:
            clingo_control.solve(
                on_model=lambda model: models.append(model.symbols(atoms=True))
            )

        models = [[str(atom) for atom in model] for model in models]
        return None, models

    def get_reasoning(self, kind, replace):
        """Extract reasoning trace from a cached response for gpt-oss-120b."""
        prompt = self.prompt[kind]
        for k in replace:
            prompt = prompt.replace(k, replace[k])
        cached = self.cache.get(kind, {}).get(prompt)
        if cached is None:
            return ""
        if self.engine == "gpt-oss-120b":
            return cached["choices"][0]["message"].get("reasoning") or ""
        return ""

    def save_mistakes(self, mistake_cols):
        df = pd.DataFrame(self.mistakes, columns=mistake_cols)
        writer = pd.ExcelWriter(self.path_mistakes)
        df.to_excel(writer, sheet_name="results")
        for col_idx in range(2, 10):
            writer.sheets["results"].set_column(col_idx, col_idx, 40)
        writer.close()
