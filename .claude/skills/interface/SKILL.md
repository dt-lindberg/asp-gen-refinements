---
name: interface
description: Documents the Streamlit evaluation app (evaluation/) — its structure, data flow from the Excel results file, rendering helpers, analytics functions, and known pitfalls. Use when modifying, debugging, or extending the interface or evaluation stats.
---

# Evaluation App

Two-page Streamlit app for inspecting pipeline results and computing aggregate evaluation metrics.

**Run:**
```bash
streamlit run evaluation/app.py
```

`interface.py` no longer exists. The old `scripts/` directory no longer exists.

---

## Directory structure

```
evaluation/
  app.py                  # Entry point: sidebar file selector + st.navigation
  eval_metrics.py         # Pure analytics functions (no Streamlit)
  hardcoded_solutions.py  # Standalone CLI script wrapping detect_hardcoded()
  pages/
    inspector.py          # Per-puzzle inspector (ported from interface.py)
    stats.py              # Aggregate evaluation stats page
```

---

## app.py — entry point

Runs before `pg.run()` on every navigation, so the sidebar code executes for both pages.

- Adds the project root to `sys.path` so pages can import `refinement_loop` etc.
- Scans `mistakes/*.xlsx` and presents a `st.selectbox` in the sidebar.
- Stores the resolved absolute path in `st.session_state["mistakes_file"]`.
- Both pages read from session state; neither loads data independently on startup.

---

## Data

Loaded via `eval_metrics.load_data(path)`:
```python
df = pd.read_excel(path, sheet_name="results", index_col=0)
df = df.reset_index(drop=True)
# Renames legacy "refinement_N" columns → "attempt_N"
```
`index_col=0` drops the integer index column pandas writes by default. `reset_index(drop=True)` gives a clean 0-based positional index.

### Relevant columns

| Column | Contents |
|---|---|
| `story`, `constraints` | Raw puzzle input |
| `constants`, `constants_formatted` | Step 1/2 data |
| `predicates` | Step 3 |
| `rules_search_space` | Step 4 |
| `constraints_paraphrased` | Step 5 |
| `rules_constraints` | Step 6 |
| `attempt_0` | Combined ASP program passed to Clingo in Step 7 (previously `refinement_0`) |
| `#answer_sets_0` | Number of answer sets from the initial Clingo run |
| `clingo_time_0` | Wall-clock time (s) of initial Clingo run |
| `clingo_errors_0` | Errors/feedback from initial Clingo run (see note below) |
| `attempt_i`, `#answer_sets_i`, `clingo_time_i`, `clingo_errors_i` | Same fields for attempt `i` (1-indexed), up to `MAX_ATTEMPTS` |
| `prediction` | Final prediction string; empty or `"N answer sets"` if incorrect |
| `solution` | Ground-truth solution |
| `reasoning_*` | LLM reasoning traces (model-dependent, may be empty) |

**`clingo_errors_i` content by case:**
- Syntax error → actual Clingo messages, e.g. `<block>:LINE:COL: error: ...`
- 0 answer sets (UNSAT) → `"0 answer sets (unsatisfiable)"` (synthesised)
- N > 1 answer sets → semantic feedback from `_build_semantic_feedback_multi` (synthesised)
- 1 answer set (success) → `""` / NaN

---

## pages/inspector.py — per-puzzle view

### Layout

```
Sidebar: file selector (app.py) + puzzle radio list
Main panel:
  Story & Constraints  (expander)
  Step 2–6             (one expander each, via show_step())
  Step 7               (show_step7)  — initial Clingo run
  Step 8               (show_step8)  — refinement attempts 1..MAX_ATTEMPTS
  Result               (show_result) — prediction vs. ground truth
```

Puzzle correctness: `is_correct(row)` — prediction is non-empty and does not match `r"\d+ answer sets"`.

### Rendering helpers

**`cell(row, col) -> str`**
Safe accessor: returns `""` for missing or NaN values, otherwise `str(val).strip()`.

**`show_text_block(label, text, height)`**
Wraps `st.text_area` (disabled, read-only). Uses a global counter `_text_block_counter` (reset to 0 each rerun) combined with `_puzzle_index` to produce unique widget keys: `p{puzzle}_tb_{n}`.

**Critical:** keys must encode the puzzle index. Without it, Streamlit reuses cached session state across puzzles, causing stale content to appear when navigating.

**`show_code_block(code, error_lines)`**
Renders ASP code as HTML with line numbers and optional ⚠ markers on error lines. Uses `st.html` — no widget state.

**`parse_error_lines(errors_str) -> set[int]`**
Extracts line numbers from Clingo error strings via `r"<block>:(\d+):"`. Only the start line of multi-line errors is highlighted.

**`show_inline_diff(before, after, error_lines)`**
ndiff-based inline diff rendered as HTML. `line_num` tracks lines in `before` (increments on `"  "` and `"- "` tags, not `"+ "`), so `error_lines` from the previous attempt align correctly.

### Key relationships

- `show_step7` reads `clingo_errors_0` — errors from running `attempt_0` through Clingo.
- `show_step8` for attempt `i` shows `clingo_errors_{i-1}` as trigger context and `clingo_errors_i` as the result.
- `MAX_ATTEMPTS` imported from `refinement_loop.py` controls how many attempt columns are expected.

---

## pages/stats.py — evaluation stats

### Sections

1. **Overview** — four `st.metric` tiles: total, correct, incorrect, potentially hardcoded.
   - Correct puzzle IDs printed inline as comma-separated `st.caption` text (1-based).
   - Hardcoded puzzle IDs printed the same way if any are flagged.

2. **Attempts to correct solution** — bar chart (Altair, green). For each correctly solved puzzle: which attempt index first yielded exactly 1 answer set. X-axis: Attempt 0 … Attempt N.

3. **Error distribution** — bar chart (Altair, blue) with a `st.multiselect` filter for attempt indices (default: all). Categories: Syntax error, Semantic unsat, Semantic multi, Correct.

4. **Average program length over attempts** — line chart with ±1 std dev band (Altair, orange). X-axis uses a numeric ordinal `attempt_num` field with `labelExpr` to avoid Altair's dot-in-field-name bug (never use `"Avg. chars:Q"` — the `.` is parsed as a nested path).

5. **Final program length by outcome** — box plot (Altair, `mark_boxplot(extent="min-max")`). One box per outcome (Solved = green, Unsolved = red). Uses `width=alt.Step(120)` to keep boxes close. Whisker caps set to `ticks=alt.MarkConfig(color="white", size=12)` for visibility on dark backgrounds.

### Altair pitfalls

- **Dot in field names**: Altair treats `.` as a nested accessor. Always use `snake_case` column names (e.g. `avg_chars`, not `Avg. chars`).
- **`use_container_width`**: deprecated after 2025-12-31. Use `width="stretch"` (was `True`) or `width="content"` (was `False`) on `st.altair_chart` and `st.dataframe`.
- **String-indexed line charts**: `st.line_chart` does not render correctly with a string index. Use Altair with a numeric ordinal axis instead.
- **Box plot whisker colour**: `rule` and `ticks` default to black; set `rule=alt.MarkConfig(color="white")` and `ticks=alt.MarkConfig(color="white", size=12)` for dark themes.

---

## eval_metrics.py — analytics functions

All functions accept a normalised DataFrame (with `attempt_N` columns). No Streamlit dependency.

| Function | Returns |
|---|---|
| `load_data(path)` | Normalised DataFrame |
| `count_correct(df)` | `(int, list[int])` — count and row indices |
| `detect_hardcoded(df)` | `dict[row_idx → list[matching_facts]]` — facts from prediction found verbatim in final ASP program |
| `attempt_distribution(df)` | `Counter[attempt_idx → count]` — for correct puzzles only |
| `error_distribution(df, attempt_filter)` | `Counter` over `syntax`, `semantic_unsat`, `semantic_multi`, `correct` |
| `avg_program_lengths(df)` | `dict[attempt_idx → {"mean": float, "std": float} | None]` |
| `program_length_by_outcome(df)` | `list[{"status": "Solved"|"Unsolved", "length": int}]` — final program per puzzle |

**Error classification logic** (used in `error_distribution`):
- `"error:"` in `clingo_errors_i` → `syntax`
- `#answer_sets_i == 0`, no syntax error → `semantic_unsat`
- `#answer_sets_i > 1` → `semantic_multi`
- `#answer_sets_i == 1` → `correct`

---

## hardcoded_solutions.py — standalone CLI

Wraps `detect_hardcoded` from `eval_metrics`. Accepts an optional filename argument:
```bash
python evaluation/hardcoded_solutions.py mistakes_0318_103804.xlsx
```
Defaults to `mistakes_0318_103804.xlsx` if no argument given.
