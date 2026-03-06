import os
import re
import time

from logger import setup_logging, get_logger
from utils import extract_code_blocks

setup_logging(log_level=os.getenv("LOG_LEVEL", "debug"))
logger = get_logger(__name__)

# The maximum number of refinement iterations
MAX_ATTEMPTS = 6
# If there are more answer sets than this, we make no attempt to show the model
SEVERELY_UNDERCONSTRAINED_THRESHOLD = 1000
MAX_VARIABLE_ATOMS = 30


def _parse_error_lines(errors):
    """Extract reported line numbers from Clingo error messages."""
    lines = set()
    for msg in errors:
        m = re.search(r"<block>:(\d+):", msg)
        if m:
            lines.add(int(m.group(1)))
    return sorted(lines)


def _annotate_with_line_numbers(code):
    """Prefix each line with its 1-based line number."""
    return "\n".join(f"{i + 1:4}: {line}" for i, line in enumerate(code.splitlines()))


def _build_error_context(code, line_numbers, window=10):
    """Extract a windowed view around each reported error line."""
    lines = code.splitlines()
    n = len(lines)
    sections = []
    for ln in line_numbers:
        lo = max(0, ln - 1 - window)
        hi = min(n, ln + window)
        block = "\n".join(
            f"{i + 1:4}{'>>>' if i + 1 == ln else '   '}: {lines[i]}"
            for i in range(lo, hi)
        )
        sections.append(f"--- Reported error at line {ln} ---\n{block}")
    return "\n\n".join(sections) if sections else "(no line numbers parsed from errors)"


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


def refinement_loop(replace, puzzle_pipeline, status, answer_sets_or_errors):
    """Refines an ASP program by looking for syntactical and 'semantic' errors.

    [Syntactical errors]
        Triggered when Clingo fails to parse a program, these errors are encoded
        as RuntimeErrors. We surface the syntax errors with approximate line numbers
        to the LLM and ask it to fix them.
    [Semantic errors]
        Triggered when compilation suceeds, but does not result in a unique answer set.
        Here too, we ask the LLM to refine the program.

    Returns:
        replace: updated replace dict
        status: final clingo status (None if no errors)
        answer_sets: final answer sets (empty if errors remain)
        attempt_data: list of MAX_ATTEMPTS tuples (code, answer_sets_count, clingo_time, clingo_errors)
    """
    attempt_data = []
    rules_all = replace["<ASP_CODE>"]

    for attempt in range(MAX_ATTEMPTS):
        # --- syntax errors ---
        if status is RuntimeError:
            # Extract the error messages
            errors = list(map(lambda x: x[1], answer_sets_or_errors))
            errors_str = "\n".join(errors)
            logger.info(
                f"Attempt {attempt + 1}/{MAX_ATTEMPTS}: syntax phase ({len(errors)} errors)"
            )

            replace["<ERRORS>"] = errors_str

            clean_code = replace["<ASP_CODE>"]
            error_lines = _parse_error_lines(errors)
            replace["<ASP_CODE>"] = _annotate_with_line_numbers(clean_code)
            replace["<ERROR_CONTEXT>"] = _build_error_context(clean_code, error_lines)

            rules_all = extract_code_blocks(
                puzzle_pipeline.gen_response("refinement_syntax", replace)
            )
            replace["<ASP_CODE>"] = rules_all

        # --- semantic errors ---
        elif status is None:
            answer_sets = answer_sets_or_errors
            n = len(answer_sets)

            # If there is one answer set, break!
            if n == 1:
                logger.info(
                    f"Attempt {attempt + 1}/{MAX_ATTEMPTS}: success (1 answer set)"
                )
                break

            # If there are no answer sets, try to make the program less constrained
            elif n == 0:
                logger.info(
                    f"Attempt {attempt + 1}/{MAX_ATTEMPTS}: semantic phase (0 answer sets)"
                )
                rules_all = extract_code_blocks(
                    puzzle_pipeline.gen_response("refinement_semantic_unsat", replace)
                )
                replace["<ASP_CODE>"] = rules_all

            # Otherwise, there are more than one answer set, in which case we try to add constraints
            else:
                logger.info(
                    f"Attempt {attempt + 1}/{MAX_ATTEMPTS}: semantic phase ({n} answer sets)"
                )
                feedback = _build_semantic_feedback_multi(answer_sets)
                replace["<SEMANTIC_FEEDBACK>"] = feedback
                replace["<NUM_ANSWER_SETS>"] = str(n)
                rules_all = extract_code_blocks(
                    puzzle_pipeline.gen_response("refinement_semantic_multi", replace)
                )
                replace["<ASP_CODE>"] = rules_all

        else:
            raise RuntimeError(f"Clingo failed with unexpected status: {status}")

        # Measure compilation time for logging
        t0 = time.time()
        status, answer_sets_or_errors = puzzle_pipeline.gen_answer_set(rules_all)
        clingo_time = round(time.time() - t0, 3)

        # Store errors for and number of answer sets for logging
        answer_sets_count = len(answer_sets_or_errors) if status is None else 0
        if status is not None:
            clingo_errors = "\n".join(x[1] for x in answer_sets_or_errors)
        elif answer_sets_count == 0:
            clingo_errors = "0 answer sets (unsatisfiable)"
        elif answer_sets_count != 1:
            clingo_errors = _build_semantic_feedback_multi(answer_sets_or_errors)
        else:
            clingo_errors = ""
        attempt_data.append((rules_all, answer_sets_count, clingo_time, clingo_errors))

    # If status is still an error after all attempts, there are no valid answer sets
    answer_sets = answer_sets_or_errors if status is None else []

    # Pad to MAX_ATTEMPTS slots for excel formatting
    # Needed when the LLM finishes before using all attempts
    while len(attempt_data) < MAX_ATTEMPTS:
        attempt_data.append(("", 0, 0.0, ""))

    return replace, status, answer_sets, attempt_data
