---
name: review-mistakes
description: Use this skill whenever the user asks to review, check, or analyse a mistakes xlsx file from the asp-gen-refinements pipeline. The files live in asp-gen-refinements/mistakes/ and contain a 'prediction' and 'solution' column that must be compared semantically. Trigger phrases "check this file", "review results", "analyse mistakes".
---

# Review Mistakes File

The pipeline generates ASP programs to solve logic-grid puzzles. Each row in a mistakes file is one puzzle. The two columns that matter are `prediction` (the answer set the model produced) and `solution` (the ground-truth tab-separated table).

## Step 1 — Extract the data

Run the following script to print every puzzle's prediction and solution side-by-side. Adjust the file path as needed.

```python
import openpyxl

wb = openpyxl.load_workbook("mistakes/<filename>.xlsx")
ws = wb.active
headers = [cell.value for cell in ws[1]]
pred_idx = headers.index("prediction")
sol_idx  = headers.index("solution")

for row in ws.iter_rows(min_row=2, values_only=True):
    puzzle_id = row[0]
    pred = row[pred_idx]
    sol  = row[sol_idx]
    print(f"--- Puzzle {puzzle_id} ---")
    print(f"PREDICTION: {repr(pred)}")
    print(f"SOLUTION:   {repr(sol)}")
    print()
```

Also extract the `#answer_sets_N` columns (one per refinement attempt, N=0,1,2,...) to identify how many answer sets the final attempt produced. These columns follow the pattern `#answer_sets_0`, `#answer_sets_1`, etc.

```python
# Collect the last non-None answer set count per row
ans_cols = [i for i, h in enumerate(headers) if h and h.startswith("#answer_sets")]

for row in ws.iter_rows(min_row=2, values_only=True):
    puzzle_id = row[0]
    counts = [row[i] for i in ans_cols if row[i] is not None]
    final_count = counts[-1] if counts else None
    print(f"Puzzle {puzzle_id}: final answer set count = {final_count}")
```

## Step 2 — Determine correctness

A prediction is **correct** if every fact in the answer set matches the solution semantically. It does not need to be syntactically identical.

### What counts as semantically equivalent

| Prediction | Solution | Verdict |
|---|---|---|
| `sighting("Iva Ingram", 4, "Tarzana")` | `August 4 \| Iva Ingram \| Tarzana` | CORRECT — 4 = August 4 |
| `assign("Fitzgerald", 1, 8)` | `Fitzgerald \| January \| 8 day` | CORRECT — 1 = January |
| `assign("Riggs", 4, 2)` | `Riggs \| April \| 2 day` | CORRECT — 4 = April |
| `team_rank_color("Spray Paints", 1, "yellow")` | `first \| Spray Paints \| yellow` | CORRECT — 1 = first |
| `match("Rob", 3, "Karen")` | `Rob \| March \| Karen` | CORRECT — 3 = March |

Key rules:
- **Integers for ordered constants are OK.** Months (1=Jan … 12=Dec), rankings (1=first, 2=second …), and day-of-month numbers are interchangeable with their string equivalents.
- **Case differences are OK.** `"Chamberlain"` matches `chamberlain`.
- **Predicate name and argument order do not matter.** Only the set of (entity, entity, entity) triples matters.
- **Extra auxiliary facts are OK** as long as the core solution triples are all present and correct (e.g., helper predicates like `diff25(...)` that are internal to the ASP encoding).
- **Fictional entity names must match exactly.** `"Versem"` ≠ `"Versum"` — different spellings of a made-up name are different entities.

### What makes a prediction wrong

- `prediction` is `None` — no answer set was produced.
- The prediction is a string like `"12 answer sets"` or `"1001 answer sets"` — clingo found multiple or no unique solutions; the pipeline could not extract a single answer.
- One or more triples are factually wrong (wrong entity pairing, wrong value, swapped assignments between rows).
- A fictional entity name is misspelled relative to the solution.

### Uncertain cases

Mark a prediction as **uncertain** if it uses integer indices for named entities (people, companies, etc.) where no natural ordering exists and the constants definition is not available to verify the mapping.

## Step 3 — Flag 1-answer-set failures

These are the most interesting failures: the model was *confident* (clingo found exactly 1 answer set, meaning the ASP program was uniquely satisfiable) but the answer set does not match the solution. This indicates a semantic modelling error — the constraints are wrong, not just unsolvable.

Identify them by combining the answer set count with correctness:

```python
# After determining correctness per puzzle, flag 1-answer-set failures
for row in ws.iter_rows(min_row=2, values_only=True):
    puzzle_id   = row[0]
    pred        = row[pred_idx]
    counts      = [row[i] for i in ans_cols if row[i] is not None]
    final_count = counts[-1] if counts else None

    # Prediction present, exactly 1 answer set, but manually determined to be wrong
    if pred is not None and final_count == 1:
        print(f"Puzzle {puzzle_id}: 1 answer set but CHECK CORRECTNESS — pred={repr(pred)}")
```

Flag these separately in the summary as **"1-AS wrong"** because they reveal constraint bugs rather than solvability bugs.

## Step 4 — Write the summary

After evaluating all puzzles, produce a summary in this format:

```
## <filename>

### Correct (N)
| Puzzle | Notes |
|--------|-------|
| **3**  | Dates 4–7 = August 4–7. All witness/town pairs match. |
...

### Uncertain (N)
| Puzzle | Notes |
| **12** | Uses integer indices for employees — mapping unverifiable without constants definition. |

### 1-Answer-Set Wrong (N)
| Puzzle | Notes |
| **X**  | 1 unique answer set produced but assignments are factually wrong (describe the mismatch). |

### Wrong (N)
List puzzle IDs inline: 0, 1, 5, 6, ...
Note any patterns (e.g., clingo overflow, empty predictions, completely wrong domain).

---
| Correct | Uncertain | 1-AS Wrong | Wrong | Accuracy |
|---------|-----------|------------|-------|----------|
| N       | N         | N          | N     | N%       |
```

When comparing multiple files, add a cross-file table showing gains and losses per puzzle ID.
