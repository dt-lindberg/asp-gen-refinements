import os
import argparse
import time
from dotenv import load_dotenv

from dataset150 import data_gen
from pipeline import Pipeline
from audit import AuditLog
from logger import setup_logging, get_logger
from utils import extract_code_blocks
from refinement_loop import (
    refinement_loop_batch,
    _build_semantic_feedback_multi,
)
from config import (
    DEFAULT_ENGINE,
    TEMPERATURE,
    MAX_TOKENS,
    MAX_ATTEMPTS,
    PROMPT_PATHS,
    SEED,
)

load_dotenv()

setup_logging(log_level=os.getenv("LOG_LEVEL", "debug"))
logger = get_logger(__name__)


def main(args):
    logger.info(f"Running with args: {args}")

    puzzle_pipeline = Pipeline(vars(args))
    puzzle_pipeline.path_prompt = PROMPT_PATHS
    seed_str = f"{args.seed:06d}"
    prefix = f"vllm_{args.engine}_seed{seed_str}"

    puzzle_pipeline.load_prompt()

    audit = AuditLog(
        audit_dir=f"audit/{prefix}",
        run_meta={
            "engine": args.engine,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "dataset_name": args.dataset_name,
            "pipeline_variant": "fewshot",
        },
    )

    num = 50 if args.num == -1 else min(args.num, 50)
    puzzles = data_gen(args.dataset_name, num)
    logger.info(f"Loaded {len(puzzles)} puzzles from {args.dataset_name}")

    # Initialize per-puzzle state
    replaces = []
    puzzle_data = []
    for i, (story, constraints, constants, solution) in enumerate(puzzles):
        puzzle_id = f"{i:03d}"
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
                "puzzle_id": puzzle_id,
                "story": story,
                "constraints": constraints,
                "constants": constants,
                "solution": solution,
                "rules_all": "",
                "answer_sets_count_0": 0,
                "clingo_time_0": 0.0,
                "clingo_errors_0": "",
            }
        )
        audit.start_puzzle(
            puzzle_id,
            inputs={
                "story": story,
                "constraints": constraints,
                "constants": constants,
                "solution": solution,
            },
        )

    # Step 1: generate complete ASP program (batch, few-shot)
    logger.info(f"Step 1: Generating ASP programs for {num} puzzles (few-shot)...")
    results = puzzle_pipeline.gen_response_batch("asp_fewshot", replaces)
    for replace, pd, (prompt, thinking, response) in zip(replaces, puzzle_data, results):
        rules_all = extract_code_blocks(response)
        pd["rules_all"] = rules_all
        replace["<ASP_CODE>"] = rules_all
        audit.record_step(
            pd["puzzle_id"], "asp_generation",
            prompt=prompt, thinking=thinking, response=response, extracted=rules_all,
        )

    # Step 2: compile + run Clingo for each puzzle (CPU, sequential)
    logger.info(f"Step 2: Running Clingo for {num} puzzles...")
    statuses = []
    asets_or_errs_list = []
    for replace, pd in zip(replaces, puzzle_data):
        rules_all = pd["rules_all"]

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

        audit.record_initial_run(
            pd["puzzle_id"],
            asp_program=rules_all,
            status="ok" if status is None else "error",
            answer_sets_count=answer_sets_count,
            clingo_time=clingo_time,
            clingo_errors=pd["clingo_errors_0"],
            answer_sets_sample=(
                [list(s) for s in answer_sets_or_errors[:5]]
                if status is None
                else []
            ),
        )

    n_correct_after_step2 = sum(
        1 for s, a in zip(statuses, asets_or_errs_list) if s is None and len(a) == 1
    )
    logger.info(
        f"After step 2: {n_correct_after_step2}/{num} puzzles solved without refinement"
    )

    # Step 3: batched refinement loop
    logger.info(f"Step 3: Starting batched refinement loop for {num} puzzles...")
    final_results = refinement_loop_batch(
        replaces, puzzle_pipeline, statuses, asets_or_errs_list,
        puzzle_data=puzzle_data, audit=audit,
        max_attempts=args.max_attempts,
    )

    # Record final outcomes
    incorrect_indices = []
    for i, (pd, (replace, status, answer_sets, attempt_data)) in enumerate(
        zip(puzzle_data, final_results)
    ):
        solved = len(answer_sets) == 1
        if not solved:
            incorrect_indices.append(i + 1)

        if solved:
            filtered_set = [fact for fact in answer_sets[0] if "," in fact]
            prediction = "\n".join(filtered_set)
        else:
            prediction = f"{len(answer_sets)} answer sets" if len(answer_sets) > 1 else ""

        audit.record_final(
            pd["puzzle_id"],
            solved=solved,
            prediction=prediction,
            ground_truth=pd["solution"],
        )

    logger.info(
        f"Number of potentially correct predictions: {num - len(incorrect_indices)}/{num}"
    )
    logger.info(f"Incorrect indices: {incorrect_indices}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="test", type=str)
    parser.add_argument("--num", default=-1, type=int)
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        type=str,
        help="engine label used for audit directory naming",
    )
    parser.add_argument("--temperature", default=TEMPERATURE, type=float)
    parser.add_argument("--max_tokens", default=MAX_TOKENS, type=int)
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--max_attempts", default=MAX_ATTEMPTS, type=int,
                        help="maximum refinement loop iterations")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
