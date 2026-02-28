"""Tab 1: Dataset Overview — stats, distributions, sparsity."""

from collections import Counter

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(
    train_dict: dict[int, list[int]],
    test_dict: dict[int, list[int]],
    n_users: int,
    n_items: int,
) -> None:
    n_train = sum(len(v) for v in train_dict.values())
    n_test = sum(len(v) for v in test_dict.values())
    density = (n_train + n_test) / (n_users * n_items) * 100

    # --- Key metrics ---
    st.subheader("Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Users", f"{n_users:,}")
    c2.metric("Items", f"{n_items:,}")
    c3.metric("Train Interactions", f"{n_train:,}")
    c4.metric("Test Interactions", f"{n_test:,}")
    c5.metric("Density", f"{density:.4f}%")

    # --- Interactions per user ---
    st.subheader("Interactions per User")
    train_per_user = [len(train_dict.get(u, [])) for u in range(n_users)]
    test_per_user = [len(test_dict.get(u, [])) for u in range(n_users)]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=train_per_user, name="Train", opacity=0.7, xbins=dict(size=5))
    )
    fig.add_trace(
        go.Histogram(x=test_per_user, name="Test", opacity=0.7, xbins=dict(size=5))
    )
    fig.update_layout(
        barmode="overlay",
        xaxis_title="# Interactions",
        yaxis_title="# Users",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Interactions per item ---
    st.subheader("Interactions per Item (Long Tail)")
    item_counts: Counter[int] = Counter()
    for items in train_dict.values():
        item_counts.update(items)
    for items in test_dict.values():
        item_counts.update(items)
    counts_per_item = [item_counts.get(i, 0) for i in range(n_items)]
    sorted_counts = sorted(counts_per_item, reverse=True)

    fig2 = px.line(
        x=list(range(len(sorted_counts))),
        y=sorted_counts,
        labels={"x": "Item rank", "y": "# Interactions"},
        log_y=True,
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # --- Top-N popular items ---
    st.subheader("Top-N Most Popular Items")
    top_n = st.slider("N", 10, 50, 20, key="topn_slider")
    most_common = item_counts.most_common(top_n)
    fig3 = px.bar(
        x=[str(item) for item, _ in most_common],
        y=[cnt for _, cnt in most_common],
        labels={"x": "Item ID", "y": "# Interactions"},
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # --- Sparsity heatmap ---
    st.subheader("Sparsity Heatmap (sampled 500×500 block)")
    rng = np.random.default_rng(42)
    sample_users = rng.choice(n_users, size=min(500, n_users), replace=False)
    sample_items = rng.choice(n_items, size=min(500, n_items), replace=False)
    sample_users.sort()
    sample_items.sort()

    user_set = set(sample_users.tolist())
    item_set = set(sample_items.tolist())
    user_idx = {u: i for i, u in enumerate(sample_users)}
    item_idx = {it: i for i, it in enumerate(sample_items)}

    mat = np.zeros((len(sample_users), len(sample_items)), dtype=np.float32)
    for u in sample_users:
        u_int = int(u)
        if u_int in train_dict:
            for it in train_dict[u_int]:
                if it in item_set:
                    mat[user_idx[u_int], item_idx[it]] = 1.0

    fig4 = px.imshow(
        mat,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"color": "Interaction"},
    )
    fig4.update_layout(
        height=500, xaxis_title="Item (sampled)", yaxis_title="User (sampled)"
    )
    st.plotly_chart(fig4, use_container_width=True)
