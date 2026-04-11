"""Inspector page — per-puzzle view of the audit trail.

Reads `audit_records` from `st.session_state` (populated by app.py).
Shows, for each LLM step and each refinement attempt, the full prompt,
thinking (when present), raw response, extracted output, and Clingo
outcome. Refinements show an inline diff against the previous program.
"""

import difflib
import re

import streamlit as st

from evaluation.eval_metrics import is_solved


STEP_ORDER = [
    ("constants_formatting", "Step 2", "Format Constants"),
    ("predicates", "Step 3", "Generate Predicates"),
    ("search_space", "Step 4", "Generate Search Space"),
    ("paraphrasing", "Step 5", "Paraphrase Constraints"),
    ("constraints", "Step 6", "Generate Constraint Rules"),
]

_text_block_counter = 0
_puzzle_index = 0


def _puzzle_label(idx, rec):
    pid = rec.get("puzzle_id", str(idx))
    final = rec.get("final") or {}
    pred = final.get("prediction", "") or ""
    if final.get("solved"):
        return f"Puzzle {pid} — correct"
    if not pred:
        return f"Puzzle {pid} — no answer set"
    return f"Puzzle {pid} — {pred}"


def _text_block(label, text, height=200):
    global _text_block_counter
    _text_block_counter += 1
    st.text_area(
        label,
        value=text or "",
        height=height,
        disabled=True,
        key=f"p{_puzzle_index}_tb_{_text_block_counter}",
    )


def _parse_error_lines(errors_str):
    if not errors_str:
        return set()
    return {int(m) for m in re.findall(r"<block>:(\d+):", errors_str)}


_CODE_DIV = (
    "display:flex;justify-content:space-between;align-items:center;"
    "white-space:pre;font-family:monospace;font-size:13px;line-height:1.4;"
    "padding:0 6px;font-variant-ligatures:none;font-feature-settings:'liga' 0;"
)


def _render_code_block(code, error_lines=None):
    error_lines = error_lines or set()
    rows = []
    for line_num, line in enumerate((code or "").splitlines(), start=1):
        escaped = (
            line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
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


def _render_inline_diff(before, after, error_lines=None):
    error_lines = error_lines or set()
    before_lines = (before or "").splitlines()
    after_lines = (after or "").splitlines()
    diff = list(difflib.ndiff(before_lines, after_lines))

    rows = []
    line_num = 0
    for entry in diff:
        tag = entry[:2]
        content = entry[2:]
        if tag == "? ":
            continue
        escaped = (
            content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
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


def _prompt_as_text(prompt):
    """Render either a string or a list of {role, content} dicts as plain text."""
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            role = msg.get("role", "?").upper()
            parts.append(f"[{role}]\n{msg.get('content', '')}")
        return "\n\n".join(parts)
    return str(prompt or "")


def _render_step(step_key, step_id, step_name, step_data):
    if not step_data:
        return
    with st.expander(f"{step_id}: {step_name}", expanded=False):
        st.markdown("**Prompt**")
        _text_block(
            f"{step_key}_prompt", _prompt_as_text(step_data.get("prompt")), height=220
        )

        thinking = step_data.get("thinking") or ""
        if thinking:
            st.markdown("**Thinking**")
            _text_block(f"{step_key}_thinking", thinking, height=220)

        st.markdown("**Response (raw)**")
        _text_block(f"{step_key}_response", step_data.get("response", ""), height=220)

        st.markdown("**Extracted**")
        _text_block(f"{step_key}_extracted", step_data.get("extracted", ""), height=220)


def _render_answer_set_sample(label_prefix, sample):
    if not sample:
        return
    text = "\n\n".join(
        f"Answer set {i + 1}:\n" + "\n".join(sorted(str(a) for a in s))
        for i, s in enumerate(sample)
    )
    _text_block(f"{label_prefix}_sample", text, height=160)


def _render_initial_run(rec):
    init = rec.get("initial_run") or {}
    if not init:
        return
    with st.expander("Step 7: Compile & Solve (Clingo)", expanded=False):
        st.markdown("**Combined ASP Program**")
        _render_code_block(
            init.get("asp_program", ""),
            error_lines=_parse_error_lines(init.get("clingo_errors", "")),
        )
        cols = st.columns(3)
        cols[0].metric("Clingo status", init.get("status", "?"))
        cols[1].metric("Answer sets", init.get("answer_sets_count", 0))
        cols[2].metric("Clingo time (s)", init.get("clingo_time", "—"))
        if init.get("clingo_errors"):
            st.markdown("**Clingo feedback**")
            _text_block("init_errors", init["clingo_errors"], height=120)
        sample = init.get("answer_sets_sample") or []
        if sample:
            st.markdown("**Answer set sample**")
            _render_answer_set_sample("init", sample)


def _render_refinements(rec):
    refs = rec.get("refinements") or []
    if not refs:
        return

    init = rec.get("initial_run") or {}
    prev_program = init.get("asp_program", "") or ""
    prev_errors = init.get("clingo_errors", "") or ""

    with st.expander(f"Step 8: Refinement ({len(refs)} attempt(s))", expanded=False):
        for i, ref in enumerate(refs, start=1):
            trigger = ref.get("trigger", "?")
            st.markdown(f"### Attempt {i} — trigger: `{trigger}`")

            if prev_errors:
                st.markdown("**Feedback that triggered this attempt**")
                _text_block(f"ref_{i}_trigger_feedback", prev_errors, height=100)

            st.markdown("**Prompt**")
            _text_block(
                f"ref_{i}_prompt", _prompt_as_text(ref.get("prompt")), height=240
            )

            thinking = ref.get("thinking") or ""
            if thinking:
                st.markdown("**Thinking**")
                _text_block(f"ref_{i}_thinking", thinking, height=220)

            st.markdown("**Response (raw)**")
            _text_block(f"ref_{i}_response", ref.get("response", ""), height=220)

            code = ref.get("extracted", "") or ""
            clingo = ref.get("clingo") or {}
            new_errors = clingo.get("clingo_errors", "") or ""

            if code and prev_program and code != prev_program:
                st.markdown("**Diff vs previous program**")
                _render_inline_diff(
                    prev_program,
                    code,
                    error_lines=_parse_error_lines(prev_errors),
                )
            elif code:
                st.markdown("**Extracted program**")
                _render_code_block(
                    code, error_lines=_parse_error_lines(new_errors)
                )

            cols = st.columns(3)
            cols[0].metric("Clingo status", clingo.get("status", "?"))
            cols[1].metric("Answer sets", clingo.get("answer_sets_count", 0))
            cols[2].metric("Clingo time (s)", clingo.get("clingo_time", "—"))
            if new_errors:
                st.markdown("**Clingo feedback**")
                _text_block(f"ref_{i}_errors", new_errors, height=100)
            sample = clingo.get("answer_sets_sample") or []
            if sample:
                st.markdown("**Answer set sample**")
                _render_answer_set_sample(f"ref_{i}", sample)

            st.divider()
            if code:
                prev_program = code
            prev_errors = new_errors


def _render_final(rec):
    final = rec.get("final") or {}
    with st.expander("Result", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Prediction**")
            pred = final.get("prediction", "") or ""
            if is_solved(rec):
                st.code(pred, language=None)
            else:
                st.warning(pred if pred else "No unique answer set")
        with cols[1]:
            st.markdown("**Ground Truth**")
            st.code(final.get("ground_truth", "") or "", language=None)


def main():
    global _puzzle_index, _text_block_counter

    records = st.session_state.get("audit_records") or []
    if not records:
        st.info("Select an audit run from the sidebar to get started.")
        return

    run_name = st.session_state.get("audit_run_name", "")
    n_correct = sum(1 for r in records if is_solved(r))
    st.caption(
        f"Run: `{run_name}` — {len(records)} puzzles, "
        f"{n_correct} correct, {len(records) - n_correct} incorrect"
    )

    with st.sidebar:
        st.divider()
        st.header("Puzzles")
        labels = [_puzzle_label(i, r) for i, r in enumerate(records)]
        selected = st.radio(
            "Select puzzle",
            options=range(len(records)),
            format_func=lambda i: labels[i],
        )

    _puzzle_index = selected
    _text_block_counter = 0
    rec = records[selected]
    pid = rec.get("puzzle_id", str(selected))

    st.subheader(f"Puzzle {pid}")

    run_meta = rec.get("run_meta") or {}
    if run_meta:
        st.caption(
            " · ".join(
                f"{k}: {v}"
                for k, v in run_meta.items()
                if k in ("engine", "seed", "dataset_name", "pipeline_variant")
            )
        )

    with st.expander("Inputs", expanded=False):
        inputs = rec.get("inputs") or {}
        _text_block("story", inputs.get("story", ""), height=150)
        _text_block("constraints", inputs.get("constraints", ""), height=120)
        _text_block("constants", inputs.get("constants", ""), height=100)

    steps = rec.get("steps") or {}
    for step_key, step_id, step_name in STEP_ORDER:
        _render_step(step_key, step_id, step_name, steps.get(step_key))

    _render_initial_run(rec)
    _render_refinements(rec)
    _render_final(rec)


main()
