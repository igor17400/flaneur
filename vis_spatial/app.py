"""Flâneur — Data Explorer: Streamlit entry point."""

from pathlib import Path

import streamlit as st

st.set_page_config(layout="wide", page_title="Flâneur — Data Explorer")

from utils import load_data  # noqa: E402

st.title("Flâneur — Gowalla Data Explorer")

_about = (Path(__file__).parent / "about.md").read_text()
with st.expander("About the Gowalla dataset & the recommendation task", expanded=False):
    st.markdown(_about)

train_dict, test_dict, n_users, n_items = load_data()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Dataset Overview", "Graph Structure", "Dataloader Inspector", "Train/Test Split"]
)

from dataloader_inspector.tab import render as render_dataloader  # noqa: E402
from dataset_overview.tab import render as render_overview  # noqa: E402
from graph_structure.tab import render as render_graph  # noqa: E402
from train_test_split.tab import render as render_split  # noqa: E402

with tab1:
    render_overview(train_dict, test_dict, n_users, n_items)
with tab2:
    render_graph(train_dict, test_dict, n_users, n_items)
with tab3:
    render_dataloader(train_dict, test_dict, n_users, n_items)
with tab4:
    render_split(train_dict, test_dict, n_users, n_items)
