"""
verify_audit.py — Re-check final.solved against ground truth content.

For each puzzle in the given audit seed directory, the existing final.solved
flag is re-evaluated: if the prediction yields exactly one answer set, the
argument values are compared against the ground truth using a three-stage
approach derived from the Bosch solver-in-the-loop repo:

  1. Full-norm set intersection (non-bijective) — ground truth values are fully
     normalised (integers extracted from decorated strings like "10 day" → "10",
     ordinal names mapped to integers, $ stripped). Each GT row must be a
     subset of some atom's argument set. Non-bijective: mirrors the approach
     in bosch/asp_handler.py.

  2. Raw substring match (non-bijective) — each GT row's minimally normalised
     values (ordinals mapped, $ stripped, spaces kept) must all appear as
     substrings in the full lowercased atom string. Handles identifier names
     that contain digits (e.g. "ISON-X42") which would be mangled by integer
     extraction in stage 1.

  3. Levenshtein fallback — pairwise edit-distance matrix with ordinal
     normalisation on both sides. A bijective 1:1 mapping must exist with all
     distances ≤ EDIT_THRESHOLD.

Only puzzles with final.solved = True are re-evaluated. Puzzles already
marked unsolved are left untouched. A copy of the audit directory is written
to <input_dir>-verified/ with the updated solved flags.

Usage:
    python verify_audit.py <audit_seed_dir>

Example:
    python verify_audit.py ../../audit/vllm_qwen3-30b-thinking_seed000013
"""

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Set, Tuple

from nltk import edit_distance

# ---- Regex patterns (from bosch/solver-in-the-loop constants.py) ----
PREDICATE_ARGUMENT_PATTERN = re.compile(r"\(([^)]*)\)")
INTEGER_REGEX_PATTERN = re.compile(r"(?<![\d.|\d:])[0-9]+(?![\d.\d:])")
TIME_REGEX_PATTERN = re.compile(r"\d+:\d+")

# Maps ordinal string constants to integers. Applied during all three stages.
# Index with lower-cased strings (from bosch/solver-in-the-loop asp_handler.py).
LEVENSHTEIN_CONSTANT_TO_INT_MAPPERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
    "sunday": 7,
}

EDIT_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Ground truth normalisation — two levels
# ---------------------------------------------------------------------------


def _strip_decorators(val: str) -> str:
    return (
        val.strip()
        .rstrip(")")
        .strip()
        .replace("$", "")
        .replace("%", "")
        .replace("'", "")
    )


def _normalize_full(val: str) -> str:
    """
    Full normalisation for stage 1 and the Levenshtein stage.

    Strips decorators, maps ordinal names to integers, handles time patterns,
    and extracts the integer component from decorated strings ("10 day" → "10",
    "100 meters" → "100"). Also collapses comma-separated thousands.
    """
    val = _strip_decorators(val)
    if re.match(r"^\d[\d,]+$", val.strip()):
        val = val.replace(",", "")
    m = TIME_REGEX_PATTERN.search(val)
    if m:
        return m.group().replace(":", "")
    lower = val.strip().lower()
    if lower in LEVENSHTEIN_CONSTANT_TO_INT_MAPPERS:
        return str(LEVENSHTEIN_CONSTANT_TO_INT_MAPPERS[lower])
    ints = INTEGER_REGEX_PATTERN.findall(val)
    if len(ints) == 1:
        return ints[0]
    if len(ints) > 1:
        return "_".join(ints)
    return lower


def _normalize_minimal(val: str) -> str:
    """
    Minimal normalisation for the raw substring stage.

    Strips decorators, maps ordinal names, handles times, but does NOT extract
    embedded integers from decorated strings. This preserves identifier names
    like "ISON-X42" intact while still normalising "January" → "1".
    """
    val = _strip_decorators(val)
    m = TIME_REGEX_PATTERN.search(val)
    if m:
        return m.group().replace(":", "")
    lower = val.strip().lower()
    if lower in LEVENSHTEIN_CONSTANT_TO_INT_MAPPERS:
        return str(LEVENSHTEIN_CONSTANT_TO_INT_MAPPERS[lower])
    return lower


def _detect_header(rows: List[str]) -> bool:
    """True when the first row looks like a column-name header (no digits, no $)."""
    return bool(rows) and not any(c.isdigit() or c == "$" for c in rows[0])


def _split_gt_row(row: str) -> List[str]:
    """Split a GT row on tab; fall back to comma if no tab found."""
    if "\t" in row:
        return row.split("\t")
    return row.split(",")


def parse_gt_rows(ground_truth: str, normalise) -> List[List[str]]:
    """
    Parse ground truth into a list of normalised value lists.

    Removes a header row if detected. Splits each remaining row on tab
    (with a comma fallback for non-standard formatting).
    """
    rows = ground_truth.split("\n")
    if _detect_header(rows):
        rows = rows[1:]
    result = []
    for row in rows:
        if not row.strip():
            continue
        vals = [normalise(v) for v in _split_gt_row(row)]
        vals = [v for v in vals if v]
        if vals:
            result.append(vals)
    return result


# ---------------------------------------------------------------------------
# Prediction parsing
# ---------------------------------------------------------------------------


def parse_prediction_atoms(prediction: str) -> List[str]:
    """Return the full lowercased atom string for each line of the prediction."""
    return [
        line.strip().lower() for line in prediction.strip().split("\n") if line.strip()
    ]


def parse_prediction_argsets(prediction: str) -> List[Set[str]]:
    """
    Parse prediction into a list of argument sets, one per atom.

    Arguments are extracted from inside the outermost parentheses, stripped of
    surrounding quotes, and lowercased.
    """
    result = []
    for line in prediction.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = PREDICATE_ARGUMENT_PATTERN.search(line)
        if not m:
            continue
        content = m.group(1).replace(", ", ",")
        args = {a.strip().strip('"').strip("'").lower() for a in content.split(",")}
        args.discard("")
        if args:
            result.append(args)
    return result


# ---------------------------------------------------------------------------
# Three-stage comparison
# ---------------------------------------------------------------------------


def _full_norm_intersection(
    pred_sets: List[Set[str]], gt_full: List[List[str]]
) -> bool:
    """
    Stage 1: non-bijective full-norm set intersection (mirrors bosch asp_handler.py).

    For each GT row (as a set), at least one prediction atom's argument set must
    be a superset. Non-bijective — the same atom may match multiple GT rows,
    but in practice puzzle values are unique enough that false positives don't occur.
    """
    gt_sets = [set(row) for row in gt_full]
    return all(any(gt <= pred for pred in pred_sets) for gt in gt_sets)


def _raw_substring(atoms: List[str], gt_minimal: List[List[str]]) -> bool:
    """
    Stage 2: non-bijective raw substring match.

    For each GT row, at least one prediction atom string must contain all the
    row's minimally-normalised values as substrings. Underscores in GT values
    are replaced with spaces before the check (consistent with bosch).
    """
    return all(
        any(all(v.replace("_", " ") in atom for v in row) for atom in atoms)
        for row in gt_minimal
    )


def _levenshtein_match(pred_sets: List[Set[str]], gt_full: List[List[str]]) -> bool:
    """
    Stage 3: pairwise edit-distance matrix with ordinal normalisation on both
    sides. Returns True if a bijective 1:1 mapping exists where every atom's
    arguments all point to the same GT row and all distances ≤ EDIT_THRESHOLD.
    """
    if len(pred_sets) != len(gt_full):
        return False
    arities = {len(p) for p in pred_sets}
    if len(arities) != 1:
        return False
    arity = arities.pop()

    pred = [list(s) for s in deepcopy(pred_sets)]
    gt = [list(row) for row in deepcopy(gt_full)]
    n = len(pred)

    def _norm(val: str) -> str:
        lower = val.lower()
        return (
            str(LEVENSHTEIN_CONSTANT_TO_INT_MAPPERS[lower])
            if lower in LEVENSHTEIN_CONSTANT_TO_INT_MAPPERS
            else val
        )

    # result_matrix[i][j] = (best_gt_row, best_gt_col, min_ed, is_contained)
    result_matrix = [[(-1, -1, float("inf"), False)] * arity for _ in range(n)]
    for i in range(n):
        for j in range(arity):
            c1 = _norm(pred[i][j])
            for m_idx in range(n):
                for k in range(len(gt[m_idx])):
                    c2 = _norm(gt[m_idx][k])
                    ed = edit_distance(c1, c2)
                    if ed > EDIT_THRESHOLD:
                        continue
                    contained = c1 in c2 or c2 in c1
                    prev = result_matrix[i][j]
                    if ed < prev[2] or (ed == prev[2] and not prev[3] and contained):
                        result_matrix[i][j] = (m_idx, k, ed, contained)

    if any(cell[0] == -1 for row in result_matrix for cell in row):
        return False
    # Bijection: every (gt_row, gt_col) pair appears exactly once
    mapping_indices = {(cell[0], cell[1]) for row in result_matrix for cell in row}
    if len(mapping_indices) < n * arity:
        return False
    # Row consistency: all args of atom i must map to the same GT row
    for row in result_matrix:
        if len({cell[0] for cell in row}) > 1:
            return False
    return True


def is_content_correct(prediction: str, ground_truth: str) -> Tuple[bool, bool]:
    """
    Run the three-stage check. Returns (is_correct, via_levenshtein).
    """
    if not prediction or not prediction.strip():
        return False, False

    pred_sets = parse_prediction_argsets(prediction)
    atoms = parse_prediction_atoms(prediction)
    gt_full = parse_gt_rows(ground_truth, _normalize_full)
    gt_minimal = parse_gt_rows(ground_truth, _normalize_minimal)

    if not pred_sets or not gt_full:
        return False, False

    # Stage 1: full-norm set intersection
    if _full_norm_intersection(pred_sets, gt_full):
        return True, False

    # Stage 2: raw substring (handles identifiers like "ISON-X42")
    if _raw_substring(atoms, gt_minimal):
        return True, False

    # Stage 3: Levenshtein
    ok = _levenshtein_match(pred_sets, gt_full)
    return ok, ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audit_dir",
        type=Path,
        help="Path to an audit seed directory (e.g., audit/vllm_qwen3-30b-thinking_seed000013)",
    )
    args = parser.parse_args()

    src_dir: Path = args.audit_dir.resolve()
    if not src_dir.is_dir():
        parser.error(f"Not a directory: {src_dir}")

    dst_dir: Path = src_dir.parent / (src_dir.name + "-verified")
    dst_dir.mkdir(exist_ok=True)

    puzzle_files = sorted(src_dir.glob("puzzle_*.json"))
    if not puzzle_files:
        print(f"No puzzle_*.json files found in {src_dir}")
        return

    n_checked = 0
    n_flipped = 0
    n_levenshtein = 0

    for puzzle_path in puzzle_files:
        with open(puzzle_path) as f:
            data = json.load(f)

        final = data.get("final", {})
        original_solved = bool(final.get("solved"))

        if not original_solved:
            # Already marked unsolved — copy as-is
            out_path = dst_dir / puzzle_path.name
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            continue

        n_checked += 1
        prediction = final.get("prediction", "")
        ground_truth = final.get("ground_truth", "")

        correct, via_lev = is_content_correct(prediction, ground_truth)

        if via_lev:
            n_levenshtein += 1

        if not correct:
            n_flipped += 1
            print(
                f"  CHANGED true → false: {puzzle_path.parent.name}/{puzzle_path.name}"
            )
            print(f"    Prediction: {repr(prediction.lower().strip())}")
            print(f"    Ground-T:   {repr(ground_truth.lower().strip())}")
            data["final"]["solved"] = False

        out_path = dst_dir / puzzle_path.name
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    print(
        f"\n{src_dir.name}: checked {n_checked} solved puzzles — "
        f"{n_flipped} flipped false, {n_levenshtein} matched via Levenshtein"
    )
    print(f"Output written to: {dst_dir}")


if __name__ == "__main__":
    main()
