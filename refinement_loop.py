import os
import time

from logger import setup_logging, get_logger
from utils import extract_code_blocks
from config import MAX_ATTEMPTS, SEVERELY_UNDERCONSTRAINED_THRESHOLD, MAX_VARIABLE_ATOMS

setup_logging(log_level=os.getenv("LOG_LEVEL", "debug"))
logger = get_logger(__name__)


def _annotate_with_line_numbers(code):
    """Prefix each line with its 1-based line number."""
    return "\n".join(f"{i + 1:4}: {line}" for i, line in enumerate(code.splitlines()))


def _build_semantic_feedback_multi(answer_sets, max_sets_shown=3):
    """Build feedback string for the under-constrained (N > 1 answer sets) case."""
    n = len(answer_sets)

    if n > SEVERELY_UNDERCONSTRAINED_THRESHOLD:
        return (
            f"The program is severely under-constrained: {n} answer sets were found. "
            "The constraints are far too weak to pin down a unique solution. "
            "Review the integrity constraints and add strong uniqueness or exclusion rules."
        )

    all_sets = [set(s) for s in answer_sets]
    universal = set.intersection(*all_sets)
    variable = sorted(set.union(*all_sets) - universal)[:MAX_VARIABLE_ATOMS]

    sample_str = "\n\n".join(
        f"Answer set {i + 1}:\n" + "\n".join(sorted(s))
        for i, s in enumerate(answer_sets[:max_sets_shown])
    )
    return (
        f"Variable atoms — present in some but not all answer sets"
        f" (up to {MAX_VARIABLE_ATOMS} shown):\n"
        + "\n".join(variable)
        + f"\n\nSample answer sets ({min(max_sets_shown, n)} of {n} shown):\n"
        + sample_str
    )


def refinement_loop_batch(
    replaces, pipeline, statuses, asets_or_errs_list, *, puzzle_data, audit,
    max_attempts=MAX_ATTEMPTS,
):
    """Batched refinement loop over all puzzles.

    Args:
        replaces: list of replace dicts (one per puzzle).
        pipeline: Pipeline instance.
        statuses: list of initial Clingo statuses (None or RuntimeError).
        asets_or_errs_list: list of initial answer_sets_or_errors.
        puzzle_data: list of puzzle dicts (used for puzzle_id lookup).
        audit: AuditLog instance to record each refinement.
        max_attempts: maximum number of refinement iterations (default: MAX_ATTEMPTS).

    Returns:
        list of (replace, status, answer_sets, attempt_data) for each puzzle,
        where attempt_data is a list of max_attempts tuples
        (code, answer_sets_count, clingo_time, clingo_errors).
    """
    n = len(replaces)
    statuses = list(statuses)
    asets_or_errs = list(asets_or_errs_list)
    attempt_data = [[] for _ in range(n)]

    # Puzzles with exactly 1 answer set on entry are already done
    done = [statuses[i] is None and len(asets_or_errs[i]) == 1 for i in range(n)]

    for attempt in range(max_attempts):
        active = [i for i in range(n) if not done[i]]
        if not active:
            logger.info(f"All {n} puzzles done after {attempt} attempts")
            break

        logger.info(
            f"Refinement attempt {attempt + 1}/{max_attempts}: {len(active)} active puzzles"
        )

        # Update replace dicts and determine prompt kind + trigger per puzzle
        kind_map = {}
        trigger_map = {}
        for i in active:
            replaces[i]["<ASP_CODE>"] = _annotate_with_line_numbers(replaces[i]["<ASP_CODE>"])
            if statuses[i] is RuntimeError:
                replaces[i]["<ERRORS>"] = "\n".join(x[1] for x in asets_or_errs[i])
                kind_map[i] = "refinement_syntax"
                trigger_map[i] = "syntax"
            elif statuses[i] is None:
                n_sets = len(asets_or_errs[i])
                if n_sets == 0:
                    kind_map[i] = "refinement_semantic_unsat"
                    trigger_map[i] = "unsat"
                else:
                    replaces[i]["<SEMANTIC_FEEDBACK>"] = _build_semantic_feedback_multi(asets_or_errs[i])
                    replaces[i]["<NUM_ANSWER_SETS>"] = str(n_sets)
                    kind_map[i] = "refinement_semantic_multi"
                    trigger_map[i] = "multi"
            else:
                raise RuntimeError(f"Puzzle {i}: unexpected Clingo status {statuses[i]}")

        # Group active puzzles by prompt kind and batch-generate
        by_kind = {}
        for i in active:
            by_kind.setdefault(kind_map[i], []).append(i)

        gen_results = {}
        for kind, indices in by_kind.items():
            kind_replaces = [replaces[i] for i in indices]
            batch = pipeline.gen_response_batch(kind, kind_replaces)
            for idx, (prompt, thinking, response) in zip(indices, batch):
                gen_results[idx] = (prompt, thinking, response)
            logger.debug(f"  Generated {len(indices)} responses for kind={kind}")

        # Run Clingo for each active puzzle and record results
        for i in active:
            prompt, thinking, response = gen_results[i]
            rules_all = extract_code_blocks(response)
            replaces[i]["<ASP_CODE>"] = rules_all

            t0 = time.time()
            status, asets_or_err = pipeline.gen_answer_set(rules_all)
            clingo_time = round(time.time() - t0, 3)

            statuses[i] = status
            asets_or_errs[i] = asets_or_err

            answer_sets_count = len(asets_or_err) if status is None else 0
            if status is not None:
                clingo_errors = "\n".join(x[1] for x in asets_or_err)
            elif answer_sets_count == 0:
                clingo_errors = "0 answer sets (unsatisfiable)"
            elif answer_sets_count != 1:
                clingo_errors = _build_semantic_feedback_multi(asets_or_err)
            else:
                clingo_errors = ""

            attempt_data[i].append((rules_all, answer_sets_count, clingo_time, clingo_errors))

            if status is None and len(asets_or_err) == 1:
                done[i] = True

            audit.record_refinement(
                puzzle_data[i]["puzzle_id"],
                trigger=trigger_map[i],
                prompt=prompt,
                thinking=thinking,
                response=response,
                extracted=rules_all,
                clingo={
                    "status": "ok" if status is None else "error",
                    "answer_sets_count": answer_sets_count,
                    "clingo_time": clingo_time,
                    "clingo_errors": clingo_errors,
                    "answer_sets_sample": (
                        [list(s) for s in asets_or_err[:5]]
                        if status is None
                        else []
                    ),
                },
            )

    n_solved = sum(done)
    logger.info(f"Refinement complete: {n_solved}/{n} puzzles solved")

    results = []
    for i in range(n):
        while len(attempt_data[i]) < max_attempts:
            attempt_data[i].append(("", 0, 0.0, ""))
        answer_sets = asets_or_errs[i] if statuses[i] is None else []
        results.append((replaces[i], statuses[i], answer_sets, attempt_data[i]))

    return results
