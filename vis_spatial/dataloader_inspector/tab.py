"""Tab 3: Dataloader Inspector — batch sampling, neg quality, triplet table."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from data import sample_negatives  # noqa: E402


def render(
    train_dict: dict[int, list[int]],
    test_dict: dict[int, list[int]],
    n_users: int,
    n_items: int,
) -> None:
    st.subheader("Batch Sampling")

    col1, col2 = st.columns(2)
    batch_size = col1.number_input(
        "Batch size", 64, 8192, 2048, step=64, key="batch_sz"
    )
    seed = col2.number_input("Random seed", 0, 9999, 42, key="dl_seed")

    if st.button("Generate Batch", key="gen_batch"):
        rng = np.random.default_rng(seed)
        users, pos_items, neg_items = sample_negatives(train_dict, n_items, rng)

        # Sample a batch
        n_total = len(users)
        idx = rng.choice(n_total, size=min(batch_size, n_total), replace=False)
        b_users = users[idx]
        b_pos = pos_items[idx]
        b_neg = neg_items[idx]

        # --- Batch statistics ---
        st.subheader("Batch Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Triplets", f"{len(b_users):,}")
        c2.metric("Unique Users", f"{len(set(b_users.tolist())):,}")
        c3.metric("Unique Pos Items", f"{len(set(b_pos.tolist())):,}")
        c4.metric("Unique Neg Items", f"{len(set(b_neg.tolist())):,}")

        # Overlap check
        pos_set = set(b_pos.tolist())
        neg_set = set(b_neg.tolist())
        overlap = pos_set & neg_set
        if overlap:
            st.warning(
                f"{len(overlap)} items appear as both positive and negative "
                f"(for different users). This is expected in BPR sampling."
            )
        else:
            st.success("No overlap between positive and negative items in this batch.")

        # --- Negative sampling quality ---
        st.subheader("Negative Sampling Quality")
        # Item popularity from training set
        item_pop: Counter[int] = Counter()
        for items in train_dict.values():
            item_pop.update(items)

        pos_popularity = [item_pop.get(int(i), 0) for i in b_pos]
        neg_popularity = [item_pop.get(int(i), 0) for i in b_neg]

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=pos_popularity,
                name="Positive items",
                opacity=0.7,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=neg_popularity,
                name="Negative items",
                opacity=0.7,
            )
        )
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Item popularity (train interactions)",
            yaxis_title="Count",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Interactive triplet table ---
        st.subheader("Triplet Table")
        df = pd.DataFrame(
            {
                "user": b_users,
                "pos_item": b_pos,
                "neg_item": b_neg,
                "pos_popularity": pos_popularity,
                "neg_popularity": neg_popularity,
            }
        )

        filter_user = st.selectbox(
            "Filter by user",
            options=["All"] + sorted(set(b_users.tolist())),
            key="filter_user",
        )
        if filter_user != "All":
            df = df[df["user"] == filter_user]

        st.dataframe(df, use_container_width=True, height=400)
