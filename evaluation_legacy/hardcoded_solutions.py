"""
Standalone script to report puzzles with hardcoded solutions.

* For library use, call detect_hardcoded(df) from eval_metrics directly
* Run as __main__ to print a report for a given mistakes file
"""

import os
import sys

# Allow running directly from this directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation_legacy.eval_metrics import detect_hardcoded, load_data

DEFAULT_MISTAKES_FILE = "mistakes_0318_103804.xlsx"


def report(path: str) -> None:
    """
    Load a mistakes file and print all puzzles flagged as potentially hardcoded.

    * Prints puzzle index, a snippet of the final ASP program, and the matching facts
    """
    df = load_data(path)
    flagged = detect_hardcoded(df)

    if not flagged:
        print("No hardcoded solutions detected.")
        return

    print(f"Found {len(flagged)} potentially hardcoded puzzle(s):\n")
    for idx, facts in flagged.items():
        # Grab final program snippet for context
        attempt_cols = sorted(
            [c for c in df.columns if c.startswith("attempt_")],
            key=lambda c: int(c.split("_")[1]),
        )
        last_col = None
        row = df.iloc[idx]
        for col in attempt_cols:
            import pandas as pd
            if pd.notna(row.get(col)) and str(row.get(col)).strip():
                last_col = col

        snippet = str(row[last_col])[:120] if last_col else "(no program)"
        print(f"Puzzle {idx}: {snippet}...")
        print(f"  Matching facts: {facts}")
        print()


if __name__ == "__main__":
    mistakes_dir = os.path.join(os.path.dirname(__file__), "..", "mistakes")
    filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MISTAKES_FILE
    path = os.path.join(mistakes_dir, filename)
    report(path)
