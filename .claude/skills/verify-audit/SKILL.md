---
name: verify-audit
description: Use this skill to run the verification script on audit seed directories, manually check every entry that gets flipped from true to false, and restore any incorrectly flipped entries in the verified JSON files.
---

# Verify Audit

Audit directories (`asp-gen-refinements/audit/`) contain per-puzzle JSON files produced by the pipeline. Each file has a `final.solved` flag. The verification script re-evaluates every `solved=true` entry against the ground truth and writes a corrected copy to a `-verified/` sibling directory. However, the script has false positives — it sometimes flips correct predictions to false. Every flip must be manually checked.

## Step 1 — Run the verification script

The script lives at `asp-gen-refinements-V2/evaluation/scripts/verify_audit.py`. Run it once per seed directory:

```bash
for seed in asp-gen-refinements/audit/vllm_qwen3-30b-thinking_seed*; do
    python asp-gen-refinements-V2/evaluation/scripts/verify_audit.py "$seed"
done
```

The script:
- Skips entries already marked `solved=false` (copies them as-is).
- Re-evaluates every `solved=true` entry using a three-stage checker (full-norm set intersection → raw substring → Levenshtein).
- Prints each flip to stdout and writes all files to `<seed_dir>-verified/`.

Capture the printed output. Each flip looks like:

```
CHANGED true → false: vllm_qwen3-30b-thinking_seed000013/puzzle_001.json
  Prediction: '...'
  Ground-T:   '...'
```

## Step 2 — Manually check every flip

For each flip, compare the `Prediction` and `Ground-T` strings printed by the script and determine whether the prediction is **semantically correct**.

The script has three known categories of false positives — cases where it incorrectly flips a correct prediction to false:

### False positive category 1: extra auxiliary facts

The model sometimes emits helper predicates alongside the solution atoms, e.g. `month_order("January",1)`, `next_month("January","February")`, `client_price("Ginger",150)`. The Levenshtein stage requires the number of prediction atoms to equal the number of GT rows, so any extra atoms cause it to fail even when the core solution atoms are all correct.

**How to spot it:** The prediction contains two or more distinct predicate names. The non-primary predicate (ordering, sequencing, pricing helpers) encodes auxiliary structure. Ignore those atoms and check only the primary predicate against the GT.

Example — correctly flipped to true:
```
Prediction: schedule("Fitzgerald","January",8)    ← primary facts
            schedule("Riggs","April",2)
            month_order("January",1)              ← auxiliary, ignore
            month_order("April",4)
GT:         Fitzgerald | January | 8 day
            Riggs      | April   | 2 day
```

### False positive category 2: integer encoding of month names

The model often encodes months as integers relative to the puzzle's month range (e.g., if the months are March–July, March=1, April=2, …, July=5). The verifier cannot match the integer `1` to the string `"march"`.

**How to spot it:** All prediction arguments are numeric where the GT has month names; the integers are consecutive and there are as many as there are months in the puzzle.

Example — correctly flipped to true:
```
Prediction: assign("Bubble Bombs",1,"Novio")      ← 1 = March
            assign("Flowcarts",2,"Gadingo")        ← 2 = April
GT:         bubble bombs | march | novio
            flowcarts    | april | gadingo
```
Verify by matching each integer to the corresponding month in sequence (smallest int = earliest month).

### False positive category 3: underscore/dash/space variants in identifier names

The model uses underscores where the GT uses dashes or spaces (or vice versa) in multi-word entity names (e.g., `DM_5000` vs `DM-5000`, `Zenix_2C` vs `Zenix 2C`, `center_back` vs `center back`). The verifier's substring stage replaces underscores in GT values with spaces before checking, but does not apply the same normalisation to prediction arguments, so hyphenated names still fail.

**How to spot it:** The prediction and GT contain the same entities but with different punctuation in compound names. All other values match.

**Important exception:** If two entities with similar names appear in the same puzzle (e.g., `center_back` and `center_forward`), check whether the prediction has them assigned to the right people, not just whether the format matches.

Example — correctly flipped to true:
```
Prediction: match("Torvia","DM_5000",625)
GT:         $625 | Torvia | DM-5000
```

### How to tell a flip is genuinely correct

A flip is correct when the prediction is factually wrong:
- Two entities have swapped assignments (e.g., wrong brand for a person, swapped positions between two players).
- Numeric values are assigned to the wrong entity (e.g., wrong year, wrong age).
- The prediction is for an entirely different puzzle than the GT (this happened consistently for `puzzle_049` across seeds — the model solved a baseball puzzle but the GT was about birds and months).
- All assignments are scrambled with no correspondence to the GT.

## Step 3 — Restore incorrectly flipped entries

For each confirmed false positive, set `final.solved = True` back in the `-verified/` JSON:

```python
import json

false_positives = {
    "asp-gen-refinements/audit/<seed_name>-verified": ["puzzle_001", "puzzle_005", ...],
    # add one entry per seed
}

for seed_dir, puzzles in false_positives.items():
    for p in puzzles:
        path = f"{seed_dir}/{p}.json"
        with open(path) as f:
            data = json.load(f)
        data["final"]["solved"] = True
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Restored: {path}")
```

Do not touch entries that are genuinely wrong — leave those as `solved=false`.

## Step 4 — Verify the final counts

After restoring, sanity-check the solved counts per seed:

```python
import json
from pathlib import Path

for verified_dir in sorted(Path("asp-gen-refinements/audit").glob("*-verified")):
    files = sorted(verified_dir.glob("puzzle_*.json"))
    solved = sum(1 for f in files if json.loads(f.read_text())["final"]["solved"])
    print(f"{verified_dir.name}: {solved}/{len(files)} solved")
```

## Reference — known false positive patterns from the thinking-model seeds

When the pipeline uses integer month encoding, it consistently uses the sequence position within the puzzle's month set (not the calendar month number). Months March–July become 1–5; March–June become 1–4; January–April become 1–4. The mapping is always: sort the GT month names chronologically, assign 1 to the earliest.

The auxiliary-fact false positives occur when the ASP program uses helper predicates to encode ordering relationships needed for the constraints (e.g., `month_order/2`, `next_month/2`, `two_months_after/2`, `client_price/2`). These are never part of the GT and always safe to ignore when evaluating correctness.
