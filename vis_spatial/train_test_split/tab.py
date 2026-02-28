"""Tab 4: Train/Test Split — split ratios, cold-start, coverage, overlap."""

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
    # --- Per-user split ratio ---
    st.subheader("Per-User Split Ratio (train / total)")
    ratios = []
    for u in range(n_users):
        tr = len(train_dict.get(u, []))
        te = len(test_dict.get(u, []))
        total = tr + te
        if total > 0:
            ratios.append(tr / total)

    fig = px.histogram(
        x=ratios,
        nbins=50,
        labels={"x": "Train ratio (train / (train + test))", "y": "# Users"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Mean split ratio: {np.mean(ratios):.4f}, Std: {np.std(ratios):.4f}")

    # --- Cold-start analysis ---
    st.subheader("Cold-Start Analysis")

    train_users = set(train_dict.keys())
    test_users = set(test_dict.keys())
    train_only_users = train_users - test_users
    test_only_users = test_users - train_users
    both_users = train_users & test_users

    train_items: set[int] = set()
    for items in train_dict.values():
        train_items.update(items)
    test_items: set[int] = set()
    for items in test_dict.values():
        test_items.update(items)
    train_only_items = train_items - test_items
    test_only_items = test_items - train_items
    both_items = train_items & test_items

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Users**")
        st.metric("Train only", f"{len(train_only_users):,}")
        st.metric("Test only (cold-start)", f"{len(test_only_users):,}")
        st.metric("In both", f"{len(both_users):,}")

    with col2:
        st.markdown("**Items**")
        st.metric("Train only", f"{len(train_only_items):,}")
        st.metric("Test only (cold-start)", f"{len(test_only_items):,}")
        st.metric("In both", f"{len(both_items):,}")

    # --- Item coverage ---
    st.subheader("Item Coverage")
    total_items_seen = len(train_items | test_items)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Items in train",
        f"{len(train_items):,} ({len(train_items) / n_items * 100:.1f}%)",
    )
    c2.metric(
        "Items in test", f"{len(test_items):,} ({len(test_items) / n_items * 100:.1f}%)"
    )
    c3.metric(
        "Total items seen",
        f"{total_items_seen:,} ({total_items_seen / n_items * 100:.1f}%)",
    )

    # Venn-style bar chart
    fig2 = go.Figure(
        go.Bar(
            x=["Train only", "Both", "Test only", "Unseen"],
            y=[
                len(train_only_items),
                len(both_items),
                len(test_only_items),
                n_items - total_items_seen,
            ],
            marker_color=["#636EFA", "#AB63FA", "#EF553B", "#CCCCCC"],
        )
    )
    fig2.update_layout(
        xaxis_title="Item set",
        yaxis_title="# Items",
        height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Per-user overlap ---
    st.subheader("Per-User Item Overlap (Train ∩ Test)")
    overlap_counts = []
    train_sizes = []
    test_sizes = []
    for u in both_users:
        tr_set = set(train_dict[u])
        te_set = set(test_dict[u])
        overlap_counts.append(len(tr_set & te_set))
        train_sizes.append(len(tr_set))
        test_sizes.append(len(te_set))

    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.histogram(
            x=overlap_counts,
            nbins=max(max(overlap_counts, default=0), 1),
            labels={"x": "# Shared items", "y": "# Users"},
            title="Items shared between train & test per user",
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        n_with_overlap = sum(1 for c in overlap_counts if c > 0)
        n_without = len(overlap_counts) - n_with_overlap
        fig4 = go.Figure(
            go.Pie(
                labels=["No overlap", "Has overlap"],
                values=[n_without, n_with_overlap],
                marker_colors=["#636EFA", "#EF553B"],
            )
        )
        fig4.update_layout(title="Users with/without item overlap", height=400)
        st.plotly_chart(fig4, use_container_width=True)
