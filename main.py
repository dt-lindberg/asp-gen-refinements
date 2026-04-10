import os
import argparse
import time
from dotenv import load_dotenv

from dataset150 import data_gen
from pipeline import Pipeline
from logger import setup_logging, get_logger
from utils import extract_code_blocks
from refinement_loop import (
    refinement_loop_batch,
    _build_semantic_feedback_multi,
)
from config import (
    MAX_ATTEMPTS,
    DEFAULT_ENGINE,
    TEMPERATURE,
    MAX_TOKENS,
    PROMPT_PATHS,
)

load_dotenv()

setup_logging(log_level=os.getenv("LOG_LEVEL", "debug"))
logger = get_logger(__name__)


def main(args):
    logger.info(f"Running with args: {args}")

    puzzle_pipeline = Pipeline(vars(args))
    puzzle_pipeline.path_prompt = PROMPT_PATHS
    prefix = "vllm_" + args.engine + "_"
    puzzle_pipeline.path_cache = {
        k: f"caches/{prefix}{k}.json" for k in puzzle_pipeline.path_prompt
    }
    logger.debug(f"Pipeline cache paths: {puzzle_pipeline.path_cache}")

    puzzle_pipeline.load_prompt()
    puzzle_pipeline.load_cache()

    num = 50 if args.num == -1 else min(args.num, 50)
    puzzles = data_gen(args.dataset_name, num)
    logger.info(f"Loaded {len(puzzles)} puzzles from {args.dataset_name}")

    # Initialize per-puzzle state
    replaces = []
    puzzle_data = []
    for story, constraints, constants, solution in puzzles:
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
        replaces.append(replace)
        puzzle_data.append(
            {
                "story": story,
                "constraints": constraints,
                "constants": constants,
                "solution": solution,
                "constraints_paraphrased": "",
                "constants_formatted": "",
                "predicates": "",
                "rules_search_space": "",
                "rules_constraints": "",
                "rules_all": "",
                "answer_sets_count_0": 0,
                "clingo_time_0": 0.0,
                "clingo_errors_0": "",
            }
        )

    try:
        # Step 2: format constants (batch)
        logger.info(f"Step 2: Formatting constants for {num} puzzles...")
        responses = puzzle_pipeline.gen_response_batch("constants", replaces)
        for replace, pd, resp in zip(replaces, puzzle_data, responses):
            cf = extract_code_blocks(resp)
            pd["constants_formatted"] = cf
            replace["<CONSTANTS>"] = cf

        # Step 3: generate predicates (batch)
        logger.info(f"Step 3: Generating predicates for {num} puzzles...")
        responses = puzzle_pipeline.gen_response_batch("predicates", replaces)
        for replace, pd, resp in zip(replaces, puzzle_data, responses):
            pred = extract_code_blocks(resp)
            pd["predicates"] = pred
            replace["<PREDICATES>"] = pred

        # Step 4: generate search space (batch)
        logger.info(f"Step 4: Generating search space for {num} puzzles...")
        responses = puzzle_pipeline.gen_response_batch("search_space", replaces)
        for replace, pd, resp in zip(replaces, puzzle_data, responses):
            pd["rules_search_space"] = extract_code_blocks(resp)

        # Step 5: paraphrase constraints (batch)
        logger.info(f"Step 5: Paraphrasing constraints for {num} puzzles...")
        responses = puzzle_pipeline.gen_response_batch("paraphrasing", replaces)
        for replace, pd, resp in zip(replaces, puzzle_data, responses):
            cp = extract_code_blocks(resp)
            pd["constraints_paraphrased"] = cp
            replace["<CONSTRAINTS>"] = cp

        # Step 6: generate constraint rules (batch)
        logger.info(f"Step 6: Generating constraint rules for {num} puzzles...")
        responses = puzzle_pipeline.gen_response_constraints_batch("constraints", replaces)
        for replace, pd, resp in zip(replaces, puzzle_data, responses):
            pd["rules_constraints"] = extract_code_blocks(resp)

    except Exception as e:
        logger.error(f"LLM batch generation failed: {e}", exc_info=True)
        logger.info("Saving partial results before exiting...")
        puzzle_pipeline.save_mistakes(
            [
                "story", "constraints", "constraints_paraphrased",
                "constants", "constants_formatted", "predicates",
                "rules_search_space", "rules_constraints",
            ]
        )
        raise

    # Step 7: compile + run Clingo for each puzzle (CPU, sequential)
    logger.info(f"Step 7: Running Clingo for {num} puzzles...")
    statuses = []
    asets_or_errs_list = []
    for replace, pd in zip(replaces, puzzle_data):
        rules_all = pd["rules_search_space"] + "\n\n" + pd["rules_constraints"]
        pd["rules_all"] = rules_all
        replace["<ASP_CODE>"] = rules_all

        t0 = time.time()
        status, answer_sets_or_errors = puzzle_pipeline.gen_answer_set(rules_all)
        clingo_time = round(time.time() - t0, 3)

        pd["clingo_time_0"] = clingo_time
        answer_sets_count = len(answer_sets_or_errors) if status is None else 0
        pd["answer_sets_count_0"] = answer_sets_count

        if status is not None:
            pd["clingo_errors_0"] = "\n".join(x[1] for x in answer_sets_or_errors)
        elif answer_sets_count == 0:
            pd["clingo_errors_0"] = "0 answer sets (unsatisfiable)"
        elif answer_sets_count != 1:
            pd["clingo_errors_0"] = _build_semantic_feedback_multi(answer_sets_or_errors)
        else:
            pd["clingo_errors_0"] = ""

        statuses.append(status)
        asets_or_errs_list.append(answer_sets_or_errors)

    n_correct_after_step7 = sum(
        1 for s, a in zip(statuses, asets_or_errs_list) if s is None and len(a) == 1
    )
    logger.info(
        f"After step 7: {n_correct_after_step7}/{num} puzzles solved without refinement"
    )

    # Step 8: batched refinement loop
    logger.info(f"Step 8: Starting batched refinement loop for {num} puzzles...")
    try:
        final_results = refinement_loop_batch(replaces, puzzle_pipeline, statuses, asets_or_errs_list)
    except Exception as e:
        logger.error(f"Refinement loop failed: {e}", exc_info=True)
        logger.info("Saving partial results before exiting...")
        puzzle_pipeline.save_mistakes(
            [
                "story", "constraints", "constraints_paraphrased",
                "constants", "constants_formatted", "predicates",
                "rules_search_space", "rules_constraints",
            ]
        )
        raise

    # Collect results and build Excel rows
    incorrect_indices = []
    for i, (pd, (replace, status, answer_sets, attempt_data)) in enumerate(
        zip(puzzle_data, final_results)
    ):
        if len(answer_sets) != 1:
            incorrect_indices.append(i + 1)

        if len(answer_sets) == 1:
            filtered_set = [fact for fact in answer_sets[0] if "," in fact]
            prediction = "\n".join(filtered_set)
        else:
            prediction = f"{len(answer_sets)} answer sets" if len(answer_sets) > 1 else ""

        if len(answer_sets) != 1 or args.debug:
            puzzle_pipeline.mistakes.append(
                (
                    pd["story"],
                    pd["constraints"],
                    pd["constraints_paraphrased"],
                    pd["constants"],
                    pd["constants_formatted"],
                    pd["predicates"],
                    pd["rules_search_space"],
                    pd["rules_constraints"],
                    # attempt 0 (original program)
                    pd["rules_all"],
                    pd["answer_sets_count_0"],
                    pd["clingo_time_0"],
                    pd["clingo_errors_0"],
                    # attempts 1..MAX_ATTEMPTS
                    *[val for quad in attempt_data for val in quad],
                    prediction,
                    pd["solution"],
                    "",  # reasoning_constants
                    "",  # reasoning_predicates
                    "",  # reasoning_search_space
                    "",  # reasoning_paraphrasing
                    "",  # reasoning_constraints
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
    parser.add_argument("--dataset_name", default="test", type=str)
    parser.add_argument("--num", default=-1, type=int)
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        type=str,
        help="engine label used for cache file naming",
    )
    parser.add_argument("--temperature", default=TEMPERATURE, type=float)
    parser.add_argument("--max_tokens", default=MAX_TOKENS, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
