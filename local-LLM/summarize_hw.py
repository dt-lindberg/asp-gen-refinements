#!/usr/bin/env python3
"""Print a concise hardware report from a monitor_hw.sh CSV."""

import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("No hardware log found.")
    sys.exit(0)

rows = []
with open(path) as f:
    for row in csv.DictReader(f):
        try:
            rows.append({k: float(v) for k, v in row.items() if k != "timestamp"})
        except ValueError:
            pass  # skip malformed rows

if not rows:
    print("Hardware log is empty.")
    sys.exit(0)


def stats(key):
    vals = [r[key] for r in rows]
    return min(vals), sum(vals) / len(vals), max(vals)


gpu_mem_total = rows[0]["gpu_mem_total_MiB"]
ram_total = rows[0]["ram_total_MiB"]
n = len(rows)

print(f"\n{'=' * 52}")
print(f"  Hardware Usage Report  ({n} samples)")
print(f"{'=' * 52}")

lo, avg, hi = stats("gpu_util_%")
print(f"  GPU utilisation  : {lo:5.1f}% / {avg:5.1f}% / {hi:5.1f}%  (min/avg/max)")

lo, avg, hi = stats("gpu_mem_used_MiB")
print(
    f"  GPU memory used  : {lo:6.0f} / {avg:6.0f} / {hi:6.0f} MiB  (of {gpu_mem_total:.0f} MiB)"
)

lo, avg, hi = stats("cpu_util_%")
print(f"  CPU utilisation  : {lo:5.1f}% / {avg:5.1f}% / {hi:5.1f}%  (min/avg/max)")

lo, avg, hi = stats("ram_used_MiB")
print(
    f"  RAM used         : {lo:6.0f} / {avg:6.0f} / {hi:6.0f} MiB  (of {ram_total:.0f} MiB)"
)

print(f"{'=' * 52}\n")
