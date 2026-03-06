import os
import argparse
import time
from dotenv import load_dotenv

from dataset150 import data_gen
from pipeline import Pipeline
from logger import setup_logging, get_logger
from utils import extract_code_blocks
from refinement_loop import (
    refinement_loop,
    MAX_ATTEMPTS,
    _build_semantic_feedback_multi,
)

load_dotenv()

setup_logging(log_level=os.getenv("LOG_LEVEL", "debug"))
logger = get_logger(__name__)


def main(args):
    logger.info(f"Running with args: {args}")

    puzzle_pipeline = Pipeline(vars(args))
    puzzle_pipeline.path_prompt = {
        "constants": "prompts/2_constant_formatting.txt",
        "predicates": "prompts/3_gen_predicates.txt",
        "search_space": "prompts/4_gen_search_space.txt",
        "paraphrasing": "prompts/5_paraphrasing.txt",
        "constraints": "prompts/6_gen_constraints.txt",
        "refinement_syntax": "prompts/7_refinement_syntax.txt",
        "refinement_semantic_unsat": "prompts/8_refinement_semantic_unsat.txt",
        "refinement_semantic_multi": "prompts/9_refinement_semantic_multi.txt",
    }
    prefix = "gpt_split_" + args.engine + "_"
    puzzle_pipeline.path_cache = {
        k: f"caches/{prefix}{k}.json" for k in puzzle_pipeline.path_prompt
    }
    logger.debug(f"Pipeline cache paths: {puzzle_pipeline.path_cache}")

    puzzle_pipeline.load_prompt()
    puzzle_pipeline.load_cache()

    num = 50 if args.num == -1 else min(args.num, 50)
    puzzles = data_gen(args.dataset_name, num)
    # # NOTE: removing first puzzle, already processed
    # puzzles = puzzles[1:]
    logger.debug(f"Loaded {len(puzzles)} puzzles from {args.dataset_name}")

    incorrect_indices = []
    logger.debug(f"Solving {num} puzzles...")

    for i in range(1, num + 1):
        logger.info(f"Solving puzzle {i}/{num}")

        story, constraints, constants, solution = puzzles[i - 1]
        replace = {
            "<STORY>": story,
            "<CONSTRAINTS>": constraints,
            "<CONSTANTS>": constants,
            "<PREDICATES>": "",
            "<ASP_CODE>": "",
            "<ERRORS>": "",
            "<ERROR_CONTEXT>": "",
            "<SEMANTIC_FEEDBACK>": "",
            "<NUM_ANSWER_SETS>": "",
        }

        # [optional: only used when skipping some steps] initialize placeholders
        constraints_paraphrased = rules_search_space = rules_constraints = rules_all = (
            answer_sets
        ) = ""

        # Step 1: extract constants and their categories
        # this step is not needed for the logic puzzles

        # Step 2: format constants and their categories
        constants_formatted = extract_code_blocks(
            puzzle_pipeline.gen_response("constants", replace)
        )
        replace["<CONSTANTS>"] = constants_formatted
        logger.info("Step 2: Formatted constants")
        logger.debug(constants_formatted)

        # Step 3: generate predicates of interest
        predicates = extract_code_blocks(
            puzzle_pipeline.gen_response("predicates", replace)
        )
        replace["<PREDICATES>"] = predicates
        logger.info("Step 3: Predicates generated")
        logger.debug(predicates)

        # Step 4: generate the search space (i.e., facts and choice rules)
        rules_search_space = extract_code_blocks(
            puzzle_pipeline.gen_response("search_space", replace)
        )
        logger.info("Step 4: Rules search space generated")
        logger.debug(rules_search_space)

        # Step 5: paraphrase the constraints in sentence format
        constraints_paraphrased = extract_code_blocks(
            puzzle_pipeline.gen_response("paraphrasing", replace)
        )
        replace["<CONSTRAINTS>"] = constraints_paraphrased
        logger.info("Step 5: Constraints paraphrased")
        logger.debug(constraints_paraphrased)

        # Step 6: generate the constraints in rule format
        # rules_constraints = puzzle_pipeline.gen_response('constraints', replace)
        rules_constraints = extract_code_blocks(
            puzzle_pipeline.gen_response_constraints("constraints", replace)
        )
        logger.info("Step 6: Rules constraints generated")
        logger.debug(rules_constraints)

        # Step 7: compute answer sets
        rules_all = rules_search_space + "\n\n" + rules_constraints
        replace["<ASP_CODE>"] = rules_all

        t0 = time.time()
        status, answer_sets_or_errors = puzzle_pipeline.gen_answer_set(rules_all)
        clingo_time_0 = round(time.time() - t0, 3)

        answer_sets_count_0 = len(answer_sets_or_errors) if status is None else 0
        if status is not None:
            clingo_errors_0 = "\n".join(x[1] for x in answer_sets_or_errors)
        elif answer_sets_count_0 == 0:
            clingo_errors_0 = "0 answer sets (unsatisfiable)"
        elif answer_sets_count_0 != 1:
            clingo_errors_0 = _build_semantic_feedback_multi(answer_sets_or_errors)
        else:
            clingo_errors_0 = ""

        # Step 8: Enter refinement loop if there are syntax errors or wrong answer set count
        needs_refinement = (status is not None) or (len(answer_sets_or_errors) != 1)
        if needs_refinement:
            replace, status, answer_sets, attempt_data = refinement_loop(
                replace, puzzle_pipeline, status, answer_sets_or_errors
            )
        else:
            answer_sets = answer_sets_or_errors
            attempt_data = [("", 0, 0.0, "")] * MAX_ATTEMPTS

        # Extract reasoning traces (gpt-oss-120b only; empty string for other models)
        reasoning_steps = {
            step: puzzle_pipeline.get_reasoning(step, replace)
            for step in (
                "constants",
                "predicates",
                "search_space",
                "paraphrasing",
                "constraints",
            )
        }

        # Step 9: evaluate final prediction
        if len(answer_sets) != 1:
            incorrect_indices.append(i)

        if len(answer_sets) == 1:
            filtered_set = [fact for fact in answer_sets[0] if "," in fact]
            prediction = "\n".join(filtered_set)
        else:
            prediction = (
                f"{len(answer_sets)} answer sets" if len(answer_sets) > 1 else ""
            )
        # record all information/mistakes
        if len(answer_sets) != 1 or args.debug:
            puzzle_pipeline.mistakes.append(
                (
                    story,
                    constraints,
                    constraints_paraphrased,
                    constants,
                    constants_formatted,
                    predicates,
                    rules_search_space,
                    rules_constraints,
                    # attempt 0 (original program)
                    rules_all,
                    answer_sets_count_0,
                    clingo_time_0,
                    clingo_errors_0,
                    # attempts 1..MAX_ATTEMPTS
                    *[val for quad in attempt_data for val in quad],
                    prediction,
                    solution,
                    reasoning_steps["constants"],
                    reasoning_steps["predicates"],
                    reasoning_steps["search_space"],
                    reasoning_steps["paraphrasing"],
                    reasoning_steps["constraints"],
                )
            )

    logger.info(
        f"Number of potentially correct predictions: {num - len(incorrect_indices)}/{num}"
    )
    logger.info(f"Incorrect indices: {incorrect_indices}")
    cols = [
        "story",
        "constraints",
        "constraints_paraphrased",
        "constants",
        "constants_formatted",
        "predicates",
        "rules_search_space",
        "rules_constraints",
        *[
            field
            for i in range(MAX_ATTEMPTS + 1)
            for field in (
                f"refinement_{i}",
                f"#answer_sets_{i}",
                f"clingo_time_{i}",
                f"clingo_errors_{i}",
            )
        ],
        "prediction",
        "solution",
        "reasoning_constants",
        "reasoning_predicates",
        "reasoning_search_space",
        "reasoning_paraphrasing",
        "reasoning_constraints",
    ]
    puzzle_pipeline.save_mistakes(cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="test",
        type=str,
        help="the engine name in \{train, test, test_HA\}",
    )
    parser.add_argument(
        "--num",
        default=-1,
        type=int,
        help="the maximum number of puzzles to evaluate; -1 means all",
    )
    parser.add_argument(
        "--step",
        default=7,
        type=int,
        help="the program will run step 1, 2, ... until this specified number",
    )
    parser.add_argument(
        "--engine",
        default="text-davinci-003",
        type=str,
        help="the engine name in \{gpt-4, text-davinci-003, text-davinci-002, text-curie-001\}",
    )
    parser.add_argument(
        "--temperature",
        default=0.0,
        type=float,
        help="the temperature for the GPT-3 model",
    )
    parser.add_argument(
        "--max_tokens",
        default=1500,
        type=int,
        help="the max number of tokens to generate",
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="debug mode"
    )
    args = parser.parse_args()
    main(args)
