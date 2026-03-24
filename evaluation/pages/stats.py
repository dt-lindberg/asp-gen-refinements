"""
Evaluation stats page — aggregate metrics over a full mistakes file.

* Reads the selected mistakes file from st.session_state["mistakes_file"]
* File selection is handled by app.py sidebar
"""

import altair as alt
import pandas as pd
import streamlit as st

from evaluation.eval_metrics import (
    MAX_ATTEMPTS,
    actual_max_attempt,
    attempt_distribution,
    avg_program_lengths,
    count_correct,
    detect_hardcoded,
    error_distribution,
    load_data,
    program_length_by_outcome,
)

# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def main():
    file_path = st.session_state.get("mistakes_file")
    if not file_path:
        st.info("Select a mistakes file from the sidebar to get started.")
        return

    try:
        df = load_data(file_path)
    except Exception as e:
        st.error(f"Failed to load {file_path}: {e}")
        return

    n_total = len(df)
    n_correct, correct_ids = count_correct(df)
    hardcoded = detect_hardcoded(df)

    # -------------------------------------------------------------------------
    # Top-level metrics
    # -------------------------------------------------------------------------

    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total puzzles", n_total)
    col2.metric("Correct", n_correct)
    col3.metric("Incorrect", n_total - n_correct)
    col4.metric("Potentially hardcoded", len(hardcoded))

    if correct_ids:
        st.caption("Correct puzzle IDs: " + ", ".join(str(i + 1) for i in correct_ids))

    if hardcoded:
        st.caption(
            "Potentially hardcoded puzzle IDs: "
            + ", ".join(str(i + 1) for i in sorted(hardcoded.keys()))
        )

    st.divider()

    # -------------------------------------------------------------------------
    # Attempt distribution (correct puzzles only)
    # -------------------------------------------------------------------------

    st.subheader("Attempts to correct solution")
    st.caption("For each correctly solved puzzle: which attempt first yielded exactly 1 answer set.")

    dist = attempt_distribution(df)
    if dist:
        max_idx = actual_max_attempt(df)
        attempt_df = pd.DataFrame(
            [{"Attempt": f"Attempt {i}", "Count": dist.get(i, 0)} for i in range(max_idx + 1)]
        )
        chart = (
            alt.Chart(attempt_df)
            .mark_bar(color="#4CAF50")
            .encode(
                x=alt.X("Attempt:N", sort=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Count:Q"),
            )
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.info("No correctly solved puzzles found.")

    st.divider()

    # -------------------------------------------------------------------------
    # Error distribution
    # -------------------------------------------------------------------------

    st.subheader("Error distribution")
    st.caption("Count of each error type across attempts. Use the filter to restrict to specific attempt indices.")

    all_attempt_indices = list(range(actual_max_attempt(df) + 1))
    selected_attempts = st.multiselect(
        "Filter by attempt index",
        options=all_attempt_indices,
        default=all_attempt_indices,
        format_func=lambda i: f"Attempt {i}",
    )

    if selected_attempts:
        err_dist = error_distribution(df, attempt_filter=selected_attempts)
        categories = [
            ("syntax", "Syntax error"),
            ("semantic_unsat", "Semantic unsat\n(0 answer sets)"),
            ("semantic_multi", "Semantic multi\n(>1 answer sets)"),
            ("correct", "Correct\n(1 answer set)"),
        ]
        err_df = pd.DataFrame(
            [{"Type": label, "Count": err_dist.get(key, 0)} for key, label in categories]
        )
        chart = (
            alt.Chart(err_df)
            .mark_bar(color="#2196F3")
            .encode(
                x=alt.X("Type:N", sort=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Count:Q"),
            )
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.info("Select at least one attempt index.")

    st.divider()

    # -------------------------------------------------------------------------
    # Average program length over attempts
    # -------------------------------------------------------------------------

    st.subheader("Average program length over attempts")
    st.caption("Mean character count of the ASP program at each attempt index (only over puzzles that reached that attempt).")

    lengths = avg_program_lengths(df)
    length_rows = [
        {
            "Attempt": f"Attempt {i}",
            "attempt_num": i,
            "avg_chars": round(lengths[i]["mean"], 1),
            "lower": round(lengths[i]["mean"] - lengths[i]["std"], 1),
            "upper": round(lengths[i]["mean"] + lengths[i]["std"], 1),
        }
        for i in range(actual_max_attempt(df) + 1)
        if lengths[i] is not None
    ]

    if length_rows:
        length_df = pd.DataFrame(length_rows)
        x_enc = alt.X(
            "attempt_num:O",
            title="Attempt",
            axis=alt.Axis(labelExpr="'Attempt ' + datum.value", labelAngle=0),
        )
        band = (
            alt.Chart(length_df)
            .mark_area(opacity=0.25, color="#FF9800")
            .encode(x=x_enc, y=alt.Y("lower:Q", title="Avg. program length (chars)"), y2="upper:Q")
        )
        line = (
            alt.Chart(length_df)
            .mark_line(point=True, color="#FF9800")
            .encode(
                x=x_enc,
                y=alt.Y("avg_chars:Q", title="Avg. program length (chars)"),
                tooltip=["Attempt:N", "avg_chars:Q", "lower:Q", "upper:Q"],
            )
        )
        st.altair_chart(band + line, width="stretch")
    else:
        st.info("No program length data available.")

    st.divider()

    # -------------------------------------------------------------------------
    # Box plot: final program length by outcome
    # -------------------------------------------------------------------------

    st.subheader("Final program length by outcome")
    st.caption("Distribution of the final ASP program length (chars) split by whether the puzzle was solved or not.")

    outcome_records = program_length_by_outcome(df)
    if outcome_records:
        outcome_df = pd.DataFrame(outcome_records)
        box = (
            alt.Chart(outcome_df, width=alt.Step(120))
            .mark_boxplot(
                extent="min-max",
                size=60,
                rule=alt.MarkConfig(color="white"),
                ticks=alt.MarkConfig(color="white", size=12),
                median=alt.MarkConfig(color="white"),
            )
            .encode(
                x=alt.X("status:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("length:Q", title="Program length (chars)"),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=["Solved", "Unsolved"],
                        range=["#4CAF50", "#F44336"],
                    ),
                    legend=None,
                ),
            )
        )
        st.altair_chart(box)
    else:
        st.info("No program length data available.")


main()
