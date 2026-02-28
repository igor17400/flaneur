"""Shared data loading for the Flâneur visualization app."""

import sys
from pathlib import Path

import streamlit as st

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import (  # noqa: E402
    _download_if_missing,
    _parse_interactions,
)


@st.cache_data
def load_data(
    data_dir: str = "data/gowalla",
) -> tuple[dict[int, list[int]], dict[int, list[int]], int, int]:
    """Load and cache the Gowalla dataset (without building JAX adjacency)."""
    root = Path(__file__).resolve().parent.parent
    full_path = str(root / data_dir)

    train_file, test_file = _download_if_missing(full_path)
    train_dict = _parse_interactions(train_file)
    test_dict = _parse_interactions(test_file)

    n_users = (
        max(max(train_dict.keys(), default=0), max(test_dict.keys(), default=0)) + 1
    )
    all_items: set[int] = set()
    for items in train_dict.values():
        all_items.update(items)
    for items in test_dict.values():
        all_items.update(items)
    n_items = max(all_items) + 1 if all_items else 0

    return train_dict, test_dict, n_users, n_items
