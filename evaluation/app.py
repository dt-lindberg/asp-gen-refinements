"""Entry point for the ASP audit-trail Streamlit app.

* Sidebar selects an audit run directory under `audit/`.
* Pages: Inspector (per-puzzle view) and Evaluation Stats (aggregate metrics).

Usage:
    streamlit run evaluation/app.py
"""

import os
import sys

import streamlit as st

# Ensure the project root is on sys.path so pages can import evaluation.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.data_loader import list_runs, load_run

st.set_page_config(page_title="ASP Pipeline (audit)", layout="wide")

audit_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "audit"
)
runs = list_runs(audit_root)

with st.sidebar:
    st.header("Run")
    if runs:
        selected = st.selectbox("Audit run", options=runs)
        run_dir = os.path.join(audit_root, selected)
        try:
            records = load_run(run_dir)
        except Exception as e:
            st.error(f"Failed to load run: {e}")
            records = []
        st.session_state["audit_records"] = records
        st.session_state["audit_run_dir"] = run_dir
        st.session_state["audit_run_name"] = selected
    else:
        st.warning(f"No audit runs found under {audit_root}")
        st.session_state["audit_records"] = []
        st.session_state["audit_run_name"] = ""

pages = [
    st.Page("pages/inspector.py", title="Inspector"),
    st.Page("pages/stats.py", title="Evaluation Stats"),
]
st.navigation(pages).run()
