"""
Inspector page — interactive per-puzzle view of ASP pipeline results.

* Reads the selected mistakes file from st.session_state["mistakes_file"]
* File selection is handled by app.py sidebar
"""

import difflib
import re

import pandas as pd
import streamlit as st

from evaluation.eval_metrics import load_data, MAX_ATTEMPTS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STEP_LABELS = {
    "constants_formatted": ("Step 2", "Format Constants"),
    "predicates": ("Step 3", "Generate Predicates"),
    "rules_search_space": ("Step 4", "Generate Search Space"),
    "constraints_paraphrased": ("Step 5", "Paraphrase Constraints"),
    "rules_constraints": ("Step 6", "Generate Constraint Rules"),
}

REASONING_COLS = {
    "constants_formatted": "reasoning_constants",
    "predicates": "reasoning_predicates",
    "rules_search_space": "reasoning_search_space",
    "constraints_paraphrased": "reasoning_paraphrasing",
    "rules_constraints": "reasoning_constraints",
}

STEP_INPUTS = {
    "constants_formatted": ["story", "constants"],
    "predicates": ["story", "constants_formatted"],
    "rules_search_space": ["constants_formatted", "predicates"],
    "constraints_paraphrased": ["story", "constraints"],
    "rules_constraints": [
        "constraints_paraphrased",
        "constants_formatted",
        "predicates",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cell(row: pd.Series, col: str) -> str:
    val = row.get(col, "")
    if pd.isna(val):
        return ""
    return str(val).strip()


def is_correct(row: pd.Series) -> bool:
    pred = cell(row, "prediction")
    return bool(pred) and not re.match(r"\d+ answer sets", pred)


def puzzle_label(idx: int, row: pd.Series) -> str:
    pred = cell(row, "prediction")
    if is_correct(row):
        return f"Puzzle {idx + 1} — correct"
    elif not pred:
        return f"Puzzle {idx + 1} — no answer set"
    else:
        return f"Puzzle {idx + 1} — {pred}"


_text_block_counter = 0
_puzzle_index = 0


def show_text_block(label: str, text: str, height: int = 200):
    global _text_block_counter
    _text_block_counter += 1
    st.text_area(
        label,
        value=text,
        height=height,
        disabled=True,
        key=f"p{_puzzle_index}_tb_{_text_block_counter}",
    )


def show_step(step_col: str, row: pd.Series):
    step_id, step_name = STEP_LABELS[step_col]
    output = cell(row, step_col)
    reasoning = cell(row, REASONING_COLS.get(step_col, ""))
    inputs = STEP_INPUTS.get(step_col, [])

    with st.expander(f"{step_id}: {step_name}", expanded=False):
        if inputs:
            st.markdown("**Input**")
            for inp in inputs:
                show_text_block(inp, cell(row, inp), height=120)

        st.markdown("**Output**")
        show_text_block(step_col, output, height=180)

        if reasoning:
            st.markdown("**Reasoning**")
            show_text_block("reasoning", reasoning, height=200)


def parse_error_lines(errors_str: str) -> set[int]:
    """Extract line numbers from Clingo error messages like '<block>:88:20-21: error: ...'"""
    return {int(m) for m in re.findall(r"<block>:(\d+):", errors_str)}


_CODE_DIV = (
    "display:flex;justify-content:space-between;align-items:center;"
    "white-space:pre;font-family:monospace;font-size:13px;line-height:1.4;"
    "padding:0 6px;font-variant-ligatures:none;font-feature-settings:'liga' 0;"
)


def show_code_block(code: str, error_lines: set[int] = None):
    """Render a code block with optional warning markers on error lines."""
    error_lines = error_lines or set()
    rows = []
    for line_num, line in enumerate(code.splitlines(), start=1):
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        warn = (
            '<span style="color:#ffaa00;user-select:none;">&#9888;</span>'
            if line_num in error_lines
            else "<span></span>"
        )
        rows.append(
            f'<div style="background:transparent;{_CODE_DIV}">'
            f'<span><span style="color:#aaa;user-select:none;margin-right:10px;">{line_num:>3}  </span>{escaped}</span>'
            f"{warn}</div>"
        )
    html = (
        '<div style="overflow-x:auto;border:1px solid #ddd;border-radius:4px;padding:4px 0">'
        + "".join(rows)
        + "</div>"
    )
    st.html(html)


def show_step7(row: pd.Series):
    rules_0 = cell(row, "attempt_0")
    clingo_time = cell(row, "clingo_time_0")
    clingo_errors = cell(row, "clingo_errors_0")
    n_sets = cell(row, "#answer_sets_0")

    with st.expander("Step 7: Compile & Solve (Clingo)", expanded=False):
        st.markdown("**Combined ASP Program**")
        show_code_block(rules_0, error_lines=parse_error_lines(clingo_errors))

        cols = st.columns(2)
        with cols[0]:
            st.metric("Clingo time (s)", clingo_time if clingo_time else "—")
        with cols[1]:
            st.metric("Answer sets", n_sets)

        if clingo_errors:
            st.markdown("**Clingo Feedback**")
            show_text_block("errors", clingo_errors, height=120)


def show_inline_diff(before: str, after: str, error_lines: set[int] = None):
    error_lines = error_lines or set()
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    diff = list(difflib.ndiff(before_lines, after_lines))

    rows = []
    line_num = 0
    for entry in diff:
        tag = entry[:2]
        content = entry[2:]
        if tag == "? ":
            continue
        escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if tag == "  ":
            bg, marker = "transparent", " "
            line_num += 1
        elif tag == "- ":
            bg, marker = "rgba(255,80,80,0.3)", "-"
            line_num += 1
        elif tag == "+ ":
            bg, marker = "rgba(80,200,80,0.3)", "+"
        else:
            continue
        warn = (
            '<span style="color:#ffaa00;user-select:none;">&#9888;</span>'
            if line_num in error_lines
            else "<span></span>"
        )
        rows.append(
            f'<div style="background:{bg};{_CODE_DIV}">'
            f'<span><span style="color:#aaa;user-select:none;margin-right:10px;">{line_num:>3} {marker}</span>{escaped}</span>'
            f"{warn}</div>"
        )

    html = (
        '<div style="overflow-x:auto;border:1px solid #ddd;border-radius:4px;padding:4px 0">'
        + "".join(rows)
        + "</div>"
    )
    st.html(html)


def show_step8(row: pd.Series):
    attempts = [
        (
            cell(row, f"attempt_{i}"),
            cell(row, f"#answer_sets_{i}"),
            cell(row, f"clingo_time_{i}"),
            cell(row, f"clingo_errors_{i}"),
        )
        for i in range(1, MAX_ATTEMPTS + 1)
    ]
    active = [(i, *data) for i, data in enumerate(attempts, 1) if any(data)]
    if not active:
        return

    all_codes = [cell(row, f"attempt_{i}") for i in range(MAX_ATTEMPTS + 1)]

    with st.expander(f"Step 8: Refinement ({len(active)} attempt(s))", expanded=False):
        for i, code, n_sets, clingo_time, clingo_errors in active:
            st.markdown(f"**Attempt {i}**")
            trigger_errors = cell(row, f"clingo_errors_{i - 1}")
            if trigger_errors:
                show_text_block(
                    f"Feedback triggering attempt {i}", trigger_errors, height=100
                )
            if code:
                prev_code = all_codes[i - 1]
                if prev_code != code:
                    st.markdown(f"**Diff (attempt {i})**")
                    show_inline_diff(
                        prev_code, code, error_lines=parse_error_lines(trigger_errors)
                    )
                else:
                    st.caption(f"Attempt {i}: no changes from previous version")
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("Clingo time (s)", clingo_time if clingo_time else "—")
            with metric_cols[1]:
                st.metric("Answer sets", n_sets if n_sets else "—")
            if clingo_errors:
                show_text_block(f"Errors from attempt {i}", clingo_errors, height=80)


def show_result(row: pd.Series):
    prediction = cell(row, "prediction")
    solution = cell(row, "solution")

    with st.expander("Result", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Prediction**")
            if not is_correct(row):
                st.warning(prediction if prediction else "No unique answer set")
            else:
                st.code(prediction, language=None)
        with cols[1]:
            st.markdown("**Ground Truth**")
            st.code(solution, language=None)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def main():
    global _puzzle_index, _text_block_counter

    file_path = st.session_state.get("mistakes_file")
    if not file_path:
        st.info("Select a mistakes file from the sidebar to get started.")
        return

    try:
        df = load_data(file_path)
    except Exception as e:
        st.error(f"Failed to load {file_path}: {e}")
        return

    n_correct = sum(1 for _, row in df.iterrows() if is_correct(row))
    st.caption(f"Loaded {len(df)} puzzles — {n_correct} correct, {len(df) - n_correct} incorrect")

    with st.sidebar:
        st.divider()
        st.header("Puzzles")
        labels = [puzzle_label(i, row) for i, row in df.iterrows()]
        selected = st.radio(
            "Select puzzle", options=range(len(df)), format_func=lambda i: labels[i]
        )

    _puzzle_index = selected
    _text_block_counter = 0
    row = df.iloc[selected]

    st.subheader(f"Puzzle {selected + 1}")

    with st.expander("Story & Constraints", expanded=False):
        show_text_block("story", cell(row, "story"), height=150)
        show_text_block("constraints (original)", cell(row, "constraints"), height=120)

    for step_col in STEP_LABELS:
        show_step(step_col, row)

    show_step7(row)
    show_step8(row)
    show_result(row)


main()
