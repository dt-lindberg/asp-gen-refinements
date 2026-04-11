"""Pure analytics functions over audit-trail records.

A "record" is the JSON dict loaded from a `puzzle_*.json` file. The
attempt-index convention is: attempt 0 = `initial_run`, attempt i>=1 =
`refinements[i - 1]`. This matches the legacy xlsx convention so the
stats page reads identically regardless of source format.

No Streamlit dependencies — safe to import anywhere.
"""

import math
from collections import Counter


def _attempt_view(attempt_dict):
    return {
        "status": attempt_dict.get("status"),
        "answer_sets_count": attempt_dict.get("answer_sets_count", 0),
        "clingo_time": attempt_dict.get("clingo_time"),
        "clingo_errors": attempt_dict.get("clingo_errors", "") or "",
    }


def iter_attempts(record):
    """Yield `(idx, view)` for initial_run followed by each refinement.

    `view` carries the clingo outcome plus the ASP program that was run
    at that attempt (`asp_program`). For refinements the program is the
    `extracted` field (which is what Clingo actually ran).
    """
    init = record.get("initial_run") or {}
    if init:
        view = _attempt_view(init)
        view["asp_program"] = init.get("asp_program", "") or ""
        yield 0, view

    for i, ref in enumerate(record.get("refinements") or [], start=1):
        clingo = ref.get("clingo") or {}
        view = _attempt_view(clingo)
        view["asp_program"] = ref.get("extracted", "") or ""
        yield i, view


def is_solved(record):
    return bool((record.get("final") or {}).get("solved"))


def count_correct(records):
    ids = [r.get("puzzle_id", "") for r in records if is_solved(r)]
    return len(ids), ids


def actual_max_attempt(records):
    max_idx = 0
    for r in records:
        n_refs = len(r.get("refinements") or [])
        # initial_run is attempt 0, refinements are 1..n_refs
        if n_refs > max_idx:
            max_idx = n_refs
    return max_idx


def attempt_distribution(records):
    """For each solved puzzle, the first attempt that produced 1 answer set."""
    dist = Counter()
    for r in records:
        if not is_solved(r):
            continue
        for idx, a in iter_attempts(r):
            if a["status"] == "ok" and a["answer_sets_count"] == 1:
                dist[idx] += 1
                break
    return dist


def error_distribution(records, attempt_filter=None):
    """Count {syntax, semantic_unsat, semantic_multi, correct} across attempts."""
    dist = Counter()
    for r in records:
        for idx, a in iter_attempts(r):
            if attempt_filter is not None and idx not in attempt_filter:
                continue
            if a["status"] == "error":
                dist["syntax"] += 1
                continue
            n_sets = a["answer_sets_count"]
            if n_sets == 0:
                dist["semantic_unsat"] += 1
            elif n_sets == 1:
                dist["correct"] += 1
            elif n_sets > 1:
                dist["semantic_multi"] += 1
    return dist


def avg_program_lengths(records):
    """Mean & std of ASP program character length per attempt index."""
    max_idx = actual_max_attempt(records)
    lengths = {i: [] for i in range(max_idx + 1)}
    for r in records:
        for idx, a in iter_attempts(r):
            program = a["asp_program"]
            if program:
                lengths[idx].append(len(program))

    result = {}
    for i, vs in lengths.items():
        if not vs:
            result[i] = None
        else:
            mean = sum(vs) / len(vs)
            std = math.sqrt(sum((x - mean) ** 2 for x in vs) / len(vs))
            result[i] = {"mean": mean, "std": std}
    return result


def program_length_by_outcome(records):
    """One record per puzzle: final program length + solved/unsolved label."""
    rows = []
    for r in records:
        last_program = ""
        for _, a in iter_attempts(r):
            if a["asp_program"]:
                last_program = a["asp_program"]
        if not last_program:
            continue
        rows.append(
            {
                "status": "Solved" if is_solved(r) else "Unsolved",
                "length": len(last_program),
            }
        )
    return rows


def detect_hardcoded(records):
    """Flag puzzles whose final program contains prediction facts verbatim."""
    results = {}
    for r in records:
        final = r.get("final") or {}
        prediction = final.get("prediction", "") or ""
        if not prediction:
            continue

        last_program = ""
        for _, a in iter_attempts(r):
            if a["asp_program"]:
                last_program = a["asp_program"]
        if not last_program:
            continue

        program_flat = last_program.replace(" ", "")
        facts = [f for f in prediction.split("\n") if f.strip()]
        matches = [f for f in facts if f in program_flat]
        if matches:
            results[r.get("puzzle_id", "")] = matches
    return results
