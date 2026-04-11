"""Per-puzzle structured audit trail.

One JSON file per puzzle at `{audit_dir}/puzzle_{id}.json`. Writes are
atomic (tmp-file + rename) and happen after every record call so the
trail on disk is always current even if the process dies mid-run.
"""

import json
import os
import threading
from datetime import datetime, timezone


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class AuditLog:
    def __init__(self, audit_dir, run_meta):
        self.audit_dir = audit_dir
        self.run_meta = dict(run_meta)
        self._lock = threading.Lock()
        self._records = {}
        os.makedirs(audit_dir, exist_ok=True)

    def _path(self, puzzle_id):
        return os.path.join(self.audit_dir, f"puzzle_{puzzle_id}.json")

    def _flush(self, puzzle_id):
        path = self._path(puzzle_id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._records[puzzle_id], f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    def start_puzzle(self, puzzle_id, inputs):
        with self._lock:
            self._records[puzzle_id] = {
                "puzzle_id": puzzle_id,
                "run_meta": dict(self.run_meta),
                "started_at": _utc_now(),
                "inputs": dict(inputs),
                "steps": {},
                "initial_run": None,
                "refinements": [],
                "final": None,
            }
            self._flush(puzzle_id)

    def record_step(self, puzzle_id, step_name, *, prompt, thinking, response, extracted):
        with self._lock:
            self._records[puzzle_id]["steps"][step_name] = {
                "prompt": prompt,
                "thinking": thinking,
                "response": response,
                "extracted": extracted,
                "timestamp": _utc_now(),
            }
            self._flush(puzzle_id)

    def record_initial_run(
        self,
        puzzle_id,
        *,
        asp_program,
        status,
        answer_sets_count,
        clingo_time,
        clingo_errors,
        answer_sets_sample,
    ):
        with self._lock:
            self._records[puzzle_id]["initial_run"] = {
                "asp_program": asp_program,
                "status": status,
                "answer_sets_count": answer_sets_count,
                "clingo_time": clingo_time,
                "clingo_errors": clingo_errors,
                "answer_sets_sample": answer_sets_sample,
                "timestamp": _utc_now(),
            }
            self._flush(puzzle_id)

    def record_refinement(
        self,
        puzzle_id,
        *,
        trigger,
        prompt,
        thinking,
        response,
        extracted,
        clingo,
    ):
        with self._lock:
            self._records[puzzle_id]["refinements"].append(
                {
                    "attempt_index": len(self._records[puzzle_id]["refinements"]) + 1,
                    "trigger": trigger,
                    "prompt": prompt,
                    "thinking": thinking,
                    "response": response,
                    "extracted": extracted,
                    "clingo": clingo,
                    "timestamp": _utc_now(),
                }
            )
            self._flush(puzzle_id)

    def record_final(self, puzzle_id, *, solved, prediction, ground_truth):
        with self._lock:
            self._records[puzzle_id]["final"] = {
                "solved": solved,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "timestamp": _utc_now(),
            }
            self._flush(puzzle_id)
