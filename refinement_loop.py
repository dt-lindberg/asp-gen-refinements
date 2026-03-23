import os
import re
import time

from logger import setup_logging, get_logger
from utils import extract_code_blocks

setup_logging(log_level=os.getenv("LOG_LEVEL", "debug"))
logger = get_logger(__name__)

# The maximum number of reattempt iterations (attempt 0 is step 7; attempts 1..MAX_ATTEMPTS are here)
MAX_ATTEMPTS = 4
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


def _build_feedback(status, asets_or_errs):
    """Format Clingo output into a feedback string for one attempt."""
    if status is RuntimeError:
        return "\n".join(x[1] for x in asets_or_errs)
    n_sets = len(asets_or_errs)
    if n_sets == 0:
        return "0 answer sets (unsatisfiable)"
    if n_sets != 1:
        return _build_semantic_feedback_multi(asets_or_errs)
    return ""


def _build_attempt_prompt(puzzle_data_i, history, system_prompt, instruction):
    """Construct the full prompt for one puzzle.

    Args:
        puzzle_data_i: dict with story, constraints_paraphrased, constants_formatted, predicates.
        history: list of (code, feedback_str) pairs, starting with attempt 0.
        system_prompt: Part A of the reattempt template.
        instruction: Part B of the reattempt template.

    Returns:
        Full prompt string.
    """
    parts = [system_prompt, ""]
    parts.append("Puzzle description:\n" + puzzle_data_i["story"])
    parts.append("\nConstraints:\n" + puzzle_data_i["constraints_paraphrased"])
    parts.append("\nConstants:\n" + puzzle_data_i["constants_formatted"])
    parts.append("\nPredicates:\n" + puzzle_data_i["predicates"])
    parts.append("")

    for idx, (code, feedback) in enumerate(history):
        n = idx + 1
        parts.append(f"<attempt_{n}>")
        parts.append("```asp")
        parts.append(code)
        parts.append("```")
        parts.append("")
        parts.append("<feedback>")
        parts.append(feedback)
        parts.append("</feedback>")
        parts.append(f"</attempt_{n}>")
        parts.append("")

    parts.append(instruction)
    return "\n".join(parts)


def multi_attempt_batch(puzzle_data, pipeline, statuses, asets_or_errs_list):
    """Multi-attempt one-shot loop over all puzzles.

    Args:
        puzzle_data: list of puzzle dicts (one per puzzle).
        pipeline: Pipeline instance.
        statuses: list of initial Clingo statuses (None or RuntimeError).
        asets_or_errs_list: list of initial answer_sets_or_errors.

    Returns:
        list of (None, status, answer_sets, attempt_data) for each puzzle,
        where attempt_data is a list of MAX_ATTEMPTS tuples
        (code, answer_sets_count, clingo_time, clingo_errors).
    """
    n = len(puzzle_data)
    statuses = list(statuses)
    asets_or_errs = list(asets_or_errs_list)
    attempt_data = [[] for _ in range(n)]

    done = [statuses[i] is None and len(asets_or_errs[i]) == 1 for i in range(n)]

    system_prompt, instruction = pipeline.prompt["reattempt"].split("===SEPARATOR===")
    system_prompt = system_prompt.strip()
    instruction = instruction.strip()

    for attempt in range(MAX_ATTEMPTS):
        active = [i for i in range(n) if not done[i]]
        if not active:
            logger.info(f"All {n} puzzles done after {attempt} attempts")
            break

        logger.info(
            f"Multi-attempt {attempt + 1}/{MAX_ATTEMPTS}: {len(active)} active puzzles"
        )

        # Build full prompts for all active puzzles
        prompts = []
        for i in active:
            history = [(puzzle_data[i]["rules_all"], puzzle_data[i]["clingo_errors_0"])]
            for code, _, _, clingo_errors in attempt_data[i]:
                history.append((code, clingo_errors))
            prompts.append(_build_attempt_prompt(puzzle_data[i], history, system_prompt, instruction))

        responses = pipeline.gen_response_raw_batch("reattempt", prompts)

        # Run Clingo for each active puzzle and record results
        for i, resp in zip(active, responses):
            rules_all = extract_code_blocks(resp)

            t0 = time.time()
            status, asets_or_err = pipeline.gen_answer_set(rules_all)
            clingo_time = round(time.time() - t0, 3)

            statuses[i] = status
            asets_or_errs[i] = asets_or_err

            answer_sets_count = len(asets_or_err) if status is None else 0
            clingo_errors = _build_feedback(status, asets_or_err)

            attempt_data[i].append((rules_all, answer_sets_count, clingo_time, clingo_errors))

            if status is None and len(asets_or_err) == 1:
                done[i] = True

    n_solved = sum(done)
    logger.info(f"Multi-attempt complete: {n_solved}/{n} puzzles solved")

    results = []
    for i in range(n):
        while len(attempt_data[i]) < MAX_ATTEMPTS:
            attempt_data[i].append(("", 0, 0.0, ""))
        answer_sets = asets_or_errs[i] if statuses[i] is None else []
        results.append((None, statuses[i], answer_sets, attempt_data[i]))

    return results
