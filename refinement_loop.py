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


def refinement_loop_batch(replaces, pipeline, statuses, asets_or_errs_list):
    """Batched refinement loop over all puzzles.

    Args:
        replaces: list of replace dicts (one per puzzle).
        pipeline: Pipeline instance.
        statuses: list of initial Clingo statuses (None or RuntimeError).
        asets_or_errs_list: list of initial answer_sets_or_errors.

    Returns:
        list of (replace, status, answer_sets, attempt_data) for each puzzle,
        where attempt_data is a list of MAX_ATTEMPTS tuples
        (code, answer_sets_count, clingo_time, clingo_errors).
    """
    n = len(replaces)
    statuses = list(statuses)
    asets_or_errs = list(asets_or_errs_list)
    attempt_data = [[] for _ in range(n)]

    # Puzzles with exactly 1 answer set on entry are already done
    done = [statuses[i] is None and len(asets_or_errs[i]) == 1 for i in range(n)]

    for attempt in range(MAX_ATTEMPTS):
        active = [i for i in range(n) if not done[i]]
        if not active:
            logger.info(f"All {n} puzzles done after {attempt} attempts")
            break

        logger.info(
            f"Refinement attempt {attempt + 1}/{MAX_ATTEMPTS}: {len(active)} active puzzles"
        )

        # Update replace dicts and determine prompt kind for each active puzzle
        kind_map = {}
        for i in active:
            replaces[i]["<ASP_CODE>"] = _annotate_with_line_numbers(replaces[i]["<ASP_CODE>"])
            if statuses[i] is RuntimeError:
                replaces[i]["<ERRORS>"] = "\n".join(x[1] for x in asets_or_errs[i])
                kind_map[i] = "refinement_syntax"
            elif statuses[i] is None:
                n_sets = len(asets_or_errs[i])
                if n_sets == 0:
                    kind_map[i] = "refinement_semantic_unsat"
                else:
                    replaces[i]["<SEMANTIC_FEEDBACK>"] = _build_semantic_feedback_multi(asets_or_errs[i])
                    replaces[i]["<NUM_ANSWER_SETS>"] = str(n_sets)
                    kind_map[i] = "refinement_semantic_multi"
            else:
                raise RuntimeError(f"Puzzle {i}: unexpected Clingo status {statuses[i]}")

        # Group active puzzles by prompt kind and batch-generate
        by_kind = {}
        for i in active:
            by_kind.setdefault(kind_map[i], []).append(i)

        new_rules = {}
        for kind, indices in by_kind.items():
            kind_replaces = [replaces[i] for i in indices]
            responses = pipeline.gen_response_batch(kind, kind_replaces)
            for idx, resp in zip(indices, responses):
                new_rules[idx] = extract_code_blocks(resp)
            logger.debug(f"  Generated {len(indices)} responses for kind={kind}")

        # Run Clingo for each active puzzle and record results
        for i in active:
            rules_all = new_rules[i]
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

    n_solved = sum(done)
    logger.info(f"Refinement complete: {n_solved}/{n} puzzles solved")

    results = []
    for i in range(n):
        while len(attempt_data[i]) < MAX_ATTEMPTS:
            attempt_data[i].append(("", 0, 0.0, ""))
        answer_sets = asets_or_errs[i] if statuses[i] is None else []
        results.append((replaces[i], statuses[i], answer_sets, attempt_data[i]))

    return results
