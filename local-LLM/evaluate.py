"""Evaluation harness for inference.py.

Contract: inference.py must write {"total_tokens", "generation_time", "tokens_per_second"}
to outputs/<name>.json.

Usage: python evaluate.py <name>
"""

import json
import sys

if len(sys.argv) != 2:
    print("Usage: evaluate.py <name>", file=sys.stderr)
    sys.exit(1)

name = sys.argv[1]
path = f"outputs/{name}.json"

with open(path) as f:
    result = json.load(f)

print(f"Total tokens:    {result['total_tokens']}")
print(f"Generation time: {result['generation_time']:.2f}s")
print(f"Tokens/s:        {result['tokens_per_second']:.2f}")
