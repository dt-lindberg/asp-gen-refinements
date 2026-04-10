---
name: review-mistakes
description: Use this skill whenever the user asks to review, check, or analyse a mistakes xlsx file from the asp-gen-refinements pipeline. The files live in asp-gen-refinements/mistakes/ and contain a 'prediction' and 'solution' column that must be compared semantically. Trigger phrases "check this file", "review results", "analyse mistakes".
---

# Review Mistakes File

The pipeline generates ASP programs to solve logic-grid puzzles. Each row in a mistakes file is one puzzle.

> **Note:** Despite the name, these files contain **all** puzzles from the run — not only the incorrect ones. Each row is a complete record including attempts, answer set counts, prediction, and ground-truth solution.

The two columns that matter are `prediction` (the answer set the model produced) and `solution` (the ground-truth tab-separated table).

## Step 1 — Extract the data

Run the following script to print every puzzle's prediction, solution, and per-attempt answer set counts. Adjust the file path as needed.

```python
import openpyxl
from collections import Counter

wb = openpyxl.load_workbook("mistakes/<filename>.xlsx")
ws = wb.active
headers = [cell.value for cell in ws[1]]
pred_idx = headers.index("prediction")
sol_idx  = headers.index("solution")

# All columns tracking how many answer sets clingo found at each attempt
ans_cols = [i for i, h in enumerate(headers) if h and h.startswith("#answer_sets")]

for row in ws.iter_rows(min_row=2, values_only=True):
    puzzle_id = row[0]
    pred = row[pred_idx]
    sol  = row[sol_idx]
    # All per-attempt counts (None means that attempt did not run)
    all_counts = [(headers[i], row[i]) for i in ans_cols]
    print(f"--- Puzzle {puzzle_id} ---")
    print(f"PREDICTION: {repr(pred)}")
    print(f"SOLUTION:   {repr(sol)}")
    print(f"AS counts:  {all_counts}")
    print()
```

### Understanding the answer set counts

Each `#answer_sets_N` column records how many answer sets clingo found at attempt N. The pipeline **extracts the prediction from the first attempt that produced exactly 1 answer set** — that is the attempt where `#answer_sets_N == 1`. Subsequent attempts will typically show 0 (because a refinement constraint was added that made the program UNSAT). This means:

- A puzzle "has 1 unique answer set" if **any** `#answer_sets_N` column equals 1.
- The `final_count` (last non-None value) is usually 0 for solved puzzles, and does **not** identify whether the prediction came from a 1-AS run.

To find all puzzles whose prediction came from a 1-answer-set run:

```python
one_as_puzzles = []
for row in ws.iter_rows(min_row=2, values_only=True):
    puzzle_id = row[0]
    counts = [row[i] for i in ans_cols if row[i] is not None]
    if any(c == 1 for c in counts):
        one_as_puzzles.append(puzzle_id)
print(f"Puzzles with 1-AS prediction: {len(one_as_puzzles)}, IDs: {one_as_puzzles}")
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

- `prediction` is `None` — no answer set was produced across all attempts.
- One or more triples are factually wrong (wrong entity pairing, wrong value, swapped assignments between rows).
- A fictional entity name is misspelled relative to the solution.

### Uncertain cases

Mark a prediction as **uncertain** if it uses integer indices for named entities (people, companies, etc.) where no natural ordering exists and the constants definition is not available to verify the mapping.

## Step 3 — Categorise by answer set outcome

Combine the correctness verdict with the answer set counts to split puzzles into four groups:

| Group | Condition | Meaning |
|-------|-----------|---------|
| **1-AS correct** | any count == 1, prediction matches solution | Model found the right unique solution |
| **1-AS wrong** | any count == 1, prediction does NOT match solution | Model was confident but wrong — constraint bug |
| **0-AS** | all counts == 0 or None, no valid prediction | Program was UNSAT throughout — never found a solution |
| **Multi-AS** | final count > 1, prediction is a count string | Program was under-constrained — too many solutions |

The **1-AS wrong** group is the most interesting for debugging: the ASP program was uniquely satisfiable but encoded the wrong constraints.

## Step 4 — Write the summary

After evaluating all puzzles, produce a summary in this format:

```
## <filename>

### 1-AS Correct (N)
| Puzzle | Notes |
|--------|-------|
| **3**  | Dates 4–7 = August 4–7. All witness/town pairs match. |

### 1-AS Wrong (N)
| Puzzle | Notes |
|--------|-------|
| **X**  | Describe the specific mismatch (e.g. Rosa↔Sherrie dates swapped). |

### Uncertain (N)
| Puzzle | Notes |
|--------|-------|
| **12** | Uses integer indices for employees — mapping unverifiable without constants definition. |

### 0-AS / No prediction (N)
List puzzle IDs inline: 0, 1, 5, 6, ...

### Multi-AS (N)
List puzzle IDs inline: 2, 4, ...

---
| 1-AS Correct | 1-AS Wrong | Uncertain | 0-AS | Multi-AS | 1-AS Accuracy |
|--------------|------------|-----------|------|----------|---------------|
| N            | N          | N         | N    | N        | N%            |
```

1-AS Accuracy = 1-AS Correct / (1-AS Correct + 1-AS Wrong).

When comparing multiple files, add a cross-file table showing gains and losses per puzzle ID.
