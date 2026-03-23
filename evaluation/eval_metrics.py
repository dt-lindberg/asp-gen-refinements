"""
Pure analytics functions for evaluating ASP pipeline results.

* All functions accept a normalised DataFrame (attempt_N columns, not refinement_N)
* No Streamlit dependencies — safe to import anywhere
"""

import re
import sys
import os
from collections import Counter

import pandas as pd

# Allow importing refinement_loop from the project root regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from refinement_loop import MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    """
    Load a mistakes Excel file and normalise column names.

    * Renames legacy 'refinement_N' columns to 'attempt_N'
    * Resets index so row positions are 0-based integers
    """
    df = pd.read_excel(path, sheet_name="results", index_col=0)
    df = df.reset_index(drop=True)
    df = df.rename(
        columns={
            c: c.replace("refinement_", "attempt_", 1)
            for c in df.columns
            if c.startswith("refinement_")
        }
    )
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cell(row: pd.Series, col: str) -> str:
    val = row.get(col, "")
    if pd.isna(val):
        return ""
    return str(val).strip()


def _is_correct(row: pd.Series) -> bool:
    pred = _cell(row, "prediction")
    return bool(pred) and not re.match(r"\d+ answer sets", pred)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def count_correct(df: pd.DataFrame) -> tuple[int, list[int]]:
    """
    Count puzzles where the pipeline produced a unique correct answer set.

    * Returns (count, list_of_row_indices)
    """
    ids = [i for i, row in df.iterrows() if _is_correct(row)]
    return len(ids), ids


def detect_hardcoded(df: pd.DataFrame) -> dict[int, list[str]]:
    """
    Flag puzzles where solution facts appear verbatim in the final ASP program.

    * Uses the last non-empty attempt column as the final program
    * Returns dict mapping row index to list of matching facts
    """
    attempt_cols = sorted(
        [c for c in df.columns if re.match(r"^attempt_\d+$", c)],
        key=lambda c: int(c.split("_")[1]),
    )

    results: dict[int, list[str]] = {}
    for idx, row in df.iterrows():
        pred = str(row.get("prediction", ""))
        if pred in ("nan", "", "None"):
            continue

        # Find the last non-empty attempt column
        last_col = None
        for col in attempt_cols:
            if pd.notna(row.get(col)) and str(row.get(col)).strip():
                last_col = col
        if last_col is None:
            continue

        # Normalise by removing spaces before substring matching
        program = str(row[last_col]).replace(" ", "")
        facts = [f for f in pred.split("\n") if f.strip()]
        matches = [f for f in facts if f in program]
        if matches:
            results[idx] = matches

    return results


def attempt_distribution(df: pd.DataFrame) -> Counter:
    """
    For each correct puzzle, find the attempt index that first yielded exactly one answer set.

    * Returns Counter mapping attempt index → number of puzzles solved at that attempt
    * Only includes puzzles that were eventually solved
    """
    dist: Counter = Counter()
    for _, row in df.iterrows():
        if not _is_correct(row):
            continue
        for i in range(MAX_ATTEMPTS + 1):
            n_str = _cell(row, f"#answer_sets_{i}")
            try:
                if int(float(n_str)) == 1:
                    dist[i] += 1
                    break
            except (ValueError, TypeError):
                continue
    return dist


def error_distribution(
    df: pd.DataFrame, attempt_filter: list[int] | None = None
) -> Counter:
    """
    Count error types across attempts, optionally restricted to specific attempt indices.

    * attempt_filter: list of attempt indices to include; None means all 0..MAX_ATTEMPTS
    * Categories: 'syntax', 'semantic_unsat', 'semantic_multi', 'correct'
    * Returns Counter
    """
    indices = attempt_filter if attempt_filter is not None else list(range(MAX_ATTEMPTS + 1))
    dist: Counter = Counter()

    for _, row in df.iterrows():
        for i in indices:
            code = _cell(row, f"attempt_{i}")
            if not code:
                # Attempt slot is empty — this puzzle didn't reach this attempt
                continue

            errors = _cell(row, f"clingo_errors_{i}")
            n_str = _cell(row, f"#answer_sets_{i}")

            try:
                n_sets = int(float(n_str)) if n_str else None
            except (ValueError, TypeError):
                n_sets = None

            if "error:" in errors:
                dist["syntax"] += 1
            elif n_sets == 0:
                dist["semantic_unsat"] += 1
            elif n_sets is not None and n_sets > 1:
                dist["semantic_multi"] += 1
            elif n_sets == 1:
                dist["correct"] += 1

    return dist


def avg_program_lengths(df: pd.DataFrame) -> dict[int, dict[str, float] | None]:
    """
    Compute mean and std deviation of ASP program character length per attempt index.

    * Returns dict mapping attempt index → {"mean": float, "std": float},
      or None if no puzzles reached that attempt
    """
    import math

    lengths: dict[int, list[int]] = {i: [] for i in range(MAX_ATTEMPTS + 1)}

    for _, row in df.iterrows():
        for i in range(MAX_ATTEMPTS + 1):
            code = _cell(row, f"attempt_{i}")
            if code:
                lengths[i].append(len(code))

    result = {}
    for i, vs in lengths.items():
        if not vs:
            result[i] = None
        else:
            mean = sum(vs) / len(vs)
            std = math.sqrt(sum((x - mean) ** 2 for x in vs) / len(vs))
            result[i] = {"mean": mean, "std": std}
    return result


def program_length_by_outcome(df: pd.DataFrame) -> list[dict]:
    """
    Return one record per puzzle with the final program length and solved/unsolved label.

    * "Final program" is the last non-empty attempt column
    * Returns list of dicts with keys: 'status' ("Solved" | "Unsolved"), 'length' (int)
    """
    attempt_cols = sorted(
        [c for c in df.columns if re.match(r"^attempt_\d+$", c)],
        key=lambda c: int(c.split("_")[1]),
    )

    records = []
    for _, row in df.iterrows():
        last_code = ""
        for col in attempt_cols:
            val = str(row.get(col, "")).strip()
            if val and val not in ("nan", "None"):
                last_code = val

        if not last_code:
            continue

        status = "Solved" if _is_correct(row) else "Unsolved"
        records.append({"status": status, "length": len(last_code)})

    return records
