"""
Entry point for the ASP pipeline Streamlit app.

* Sidebar handles mistakes file selection (shared across all pages)
* Pages: Inspector (per-puzzle view) and Evaluation Stats (aggregate metrics)

Usage:
    streamlit run evaluation/app.py
"""

import glob
import os
import sys

import streamlit as st

# Ensure the project root is on the path so pages can import refinement_loop etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="ASP Pipeline", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: shared file selector
# ---------------------------------------------------------------------------

mistakes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mistakes")
available = sorted(glob.glob(os.path.join(mistakes_dir, "*.xlsx")), reverse=True)
file_labels = [os.path.basename(f) for f in available]

with st.sidebar:
    st.header("Data")
    if available:
        selected_label = st.selectbox("Mistakes file", options=file_labels)
        st.session_state["mistakes_file"] = available[file_labels.index(selected_label)]
    else:
        st.warning("No .xlsx files found in mistakes/")
        st.session_state["mistakes_file"] = None

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

pages = [
    st.Page("pages/inspector.py", title="Inspector"),
    st.Page("pages/stats.py", title="Evaluation Stats"),
]

pg = st.navigation(pages)
pg.run()
