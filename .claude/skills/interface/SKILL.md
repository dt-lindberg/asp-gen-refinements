---
name: interface
description: Documents the Streamlit inspector app (interface.py) — its structure, data flow from the Excel results file, rendering helpers, and known pitfalls. Use when modifying, debugging, or extending the interface.
---

# Interface (interface.py)

Streamlit app for inspecting pipeline results stored in an Excel file.

**Run:**
```bash
streamlit run interface.py -- --file mistakes/mistakes_MMDD_HHMMSS.xlsx
```
Defaults to `mistakes.xlsx` if `--file` is omitted.

---

## Data

Loaded once per rerun from the Excel sheet `"results"`:
```python
df = pd.read_excel(path, sheet_name="results", index_col=0)
df = df.reset_index(drop=True)
```
`index_col=0` drops the integer index column that pandas writes by default. `reset_index(drop=True)` gives a clean 0-based positional index.

### Relevant columns

| Column | Contents |
|---|---|
| `story`, `constraints` | Raw puzzle input |
| `constants`, `constants_formatted` | Step 1/2 data |
| `predicates` | Step 3 |
| `rules_search_space` | Step 4 |
| `constraints_paraphrased` | Step 5 |
| `rules_constraints` | Step 6 |
| `refinement_0` | Combined ASP program (`rules_search_space + "\n\n" + rules_constraints`) passed to Clingo in Step 7 |
| `#answer_sets_0` | Number of answer sets from the initial Clingo run |
| `clingo_time_0` | Wall-clock time (s) of initial Clingo run |
| `clingo_errors_0` | Errors/feedback from initial Clingo run (see note below) |
| `refinement_i`, `#answer_sets_i`, `clingo_time_i`, `clingo_errors_i` | Same fields for refinement attempt `i` (1-indexed), up to `MAX_ATTEMPTS` |
| `prediction` | Final prediction string; empty or `"N answer sets"` if incorrect |
| `solution` | Ground-truth solution |
| `reasoning_*` | LLM reasoning traces (model-dependent, may be empty) |

**`clingo_errors_0` content by case:**
- Syntax error → actual Clingo messages, e.g. `<block>:LINE:COL: error: ...`
- 0 answer sets (UNSAT) → `"0 answer sets (unsatisfiable)"` (synthesised)
- N > 1 answer sets → semantic feedback from `_build_semantic_feedback_multi` (synthesised)
- 1 answer set (success) → `""` / NaN

---

## Layout

```
Sidebar: puzzle radio list  →  selected index → df.iloc[selected]

Main panel:
  Story & Constraints  (expander)
  Step 2–6             (one expander each, via show_step())
  Step 7               (show_step7)  — initial Clingo run
  Step 8               (show_step8)  — refinement attempts 1..MAX_ATTEMPTS
  Result               (show_result) — prediction vs. ground truth
```

Puzzle correctness is determined by `is_correct(row)`: prediction is non-empty and does not match `r"\d+ answer sets"`.

---

## Rendering helpers

### `cell(row, col) -> str`
Safe accessor: returns `""` for missing or NaN values, otherwise `str(val).strip()`.

### `show_text_block(label, text, height)`
Wraps `st.text_area` (disabled, read-only). Uses a global counter `_text_block_counter` (reset to 0 each rerun) combined with `_puzzle_index` (set to `selected` before rendering) to produce unique widget keys: `p{puzzle}_tb_{n}`.

**Critical:** keys must encode the puzzle index. Without it, Streamlit reuses cached session state across puzzles, causing stale content to appear when navigating between puzzles that have errors.

### `show_code_block(code, error_lines)`
Renders an ASP program as HTML with line numbers and optional ⚠ markers on error lines. No Streamlit widget state — uses `st.html`.

### `parse_error_lines(errors_str) -> set[int]`
Extracts line numbers from Clingo error strings via `r"<block>:(\d+):"`. Returns a set of start-line numbers. Multi-line errors (e.g. `<block>:8:1-11:53:`) only highlight the start line.

### `show_inline_diff(before, after, error_lines)`
ndiff-based inline diff rendered as HTML. `line_num` tracks lines in `before` (increments on `"  "` and `"- "` tags, not `"+ "`), so `error_lines` from the previous attempt's clingo errors align correctly.

---

## Key relationships

- `show_step7` reads `clingo_errors_0` — errors from running `refinement_0` through Clingo.
- `show_step8` for attempt `i` shows `clingo_errors_{i-1}` as *trigger* context and `clingo_errors_i` as the result of that attempt.
- `MAX_ATTEMPTS` is imported from `refinement_loop.py` and controls how many refinement columns are expected. If `MAX_ATTEMPTS = 0`, step 8 is hidden entirely.
