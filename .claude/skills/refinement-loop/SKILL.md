---
name: refinement-loop
description: Describes the architecture and logic of the two-phase refinement loop split across main.py and refinement_loop.py.
---

## Overview

Step 8 of the pipeline refines ASP code when Clingo reports errors or produces the wrong number of answer sets. The logic is split:
- `main.py` — owns the initial `gen_answer_set` call and guards entry into the loop
- `refinement_loop.py` — owns the retry loop and returns resolved state

Unicode errors are prevented upstream: the program is ASCII-encoded before being passed to Clingo (`pipeline.py:gen_answer_set`), so `UnicodeDecodeError` is no longer a possible status.

## Flow

```
main.py: gen_answer_set(rules_all) → (status, answer_sets_or_errors)

needs_refinement = status is not None or len(answer_sets_or_errors) != 1
if needs_refinement:
    → refinement_loop(...)
else:
    → answer_sets = answer_sets_or_errors
      attempt_data = [("", 0, 0.0, "")] * MAX_ATTEMPTS
```

`refinement_loop` runs up to `MAX_ATTEMPTS` iterations under a shared budget:

```
for attempt in range(MAX_ATTEMPTS):
    if status is RuntimeError:          # Phase 1: syntax error
        → refinement_syntax prompt
    elif status is None:
        if len(answer_sets) == 1: break # success
        elif len(answer_sets) == 0:     # Phase 2a: over-constrained
            → refinement_semantic_unsat prompt
        else:                           # Phase 2b: under-constrained
            → refinement_semantic_multi prompt
    else:
        raise RuntimeError(unexpected status)
    gen_answer_set(rules_all)  # re-evaluate after each attempt
```

## Return contract

`refinement_loop` returns:
```python
(replace, status, answer_sets, attempt_data)
```

- `answer_sets` — resolved: either the answer sets list or `[]` if all attempts failed
- `attempt_data` — list of `MAX_ATTEMPTS` tuples `(rules_all, answer_sets_count, clingo_time, clingo_errors)`, always padded to `MAX_ATTEMPTS` slots for consistent Excel output

## Key design decisions

- **Guard in main.py** (`needs_refinement`): triggers on both syntax errors (`status is not None`) and wrong answer set count.
- **Single budget**: `MAX_ATTEMPTS` is shared across Phase 1 and Phase 2 — no separate counters.
- **Semantic feedback**: for N > 1 answer sets, `_build_semantic_feedback_multi` computes variable atoms and sample answer sets; for N > `SEVERELY_UNDERCONSTRAINED_THRESHOLD` only the count is surfaced.
- **`reasoning_steps` lives in main.py**, not in `refinement_loop` — unrelated to error refinement.
- **`MAX_ATTEMPTS` is imported into main.py** so Excel column generation stays in sync automatically.
