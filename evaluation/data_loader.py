"""Load audit-trail records produced by the pipeline.

Each run of the pipeline writes one JSON file per puzzle at
`audit/<run_name>/puzzle_<id>.json`. This module discovers runs and
loads their puzzle records as plain dicts.
"""

import glob
import json
import os


def list_runs(audit_root):
    """Return run directory names under `audit_root`, most recent first."""
    if not os.path.isdir(audit_root):
        return []
    entries = [
        d
        for d in os.listdir(audit_root)
        if os.path.isdir(os.path.join(audit_root, d))
    ]
    return sorted(entries, reverse=True)


def load_run(run_dir):
    """Load every `puzzle_*.json` in `run_dir` sorted by filename."""
    paths = sorted(glob.glob(os.path.join(run_dir, "puzzle_*.json")))
    records = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records
