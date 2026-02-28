"""Tab 2: Graph Structure — degree distributions, adjacency, subgraph."""

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
    # Compute degrees
    user_degrees = np.array([len(train_dict.get(u, [])) for u in range(n_users)])
    item_degree_counter: Counter[int] = Counter()
    for items in train_dict.values():
        item_degree_counter.update(items)
    item_degrees = np.array([item_degree_counter.get(i, 0) for i in range(n_items)])

    # --- Stats ---
    st.subheader("Graph Statistics (Training Set)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg User Degree", f"{user_degrees.mean():.2f}")
    c2.metric("Max User Degree", f"{user_degrees.max()}")
    c3.metric("Avg Item Degree", f"{item_degrees.mean():.2f}")
    c4.metric("Max Item Degree", f"{item_degrees.max()}")

    active_users = int((user_degrees > 0).sum())
    active_items = int((item_degrees > 0).sum())
    st.info(
        f"Active nodes: {active_users:,} users, {active_items:,} items "
        f"(isolated: {n_users - active_users:,} users, {n_items - active_items:,} items)"
    )

    # --- Degree distribution (log-log) ---
    st.subheader("Degree Distribution (log-log)")
    col1, col2 = st.columns(2)

    with col1:
        user_deg_counts = Counter(user_degrees[user_degrees > 0].tolist())
        fig_u = px.scatter(
            x=list(user_deg_counts.keys()),
            y=list(user_deg_counts.values()),
            log_x=True,
            log_y=True,
            labels={"x": "Degree", "y": "Count"},
            title="User Degree Distribution",
        )
        fig_u.update_layout(height=400)
        st.plotly_chart(fig_u, use_container_width=True)

    with col2:
        item_deg_counts = Counter(item_degrees[item_degrees > 0].tolist())
        fig_i = px.scatter(
            x=list(item_deg_counts.keys()),
            y=list(item_deg_counts.values()),
            log_x=True,
            log_y=True,
            labels={"x": "Degree", "y": "Count"},
            title="Item Degree Distribution",
        )
        fig_i.update_layout(height=400)
        st.plotly_chart(fig_i, use_container_width=True)

    # --- Adjacency spy plot ---
    st.subheader("Adjacency Matrix Spy Plot (sampled subgraph)")
    spy_size = st.slider("Subgraph size", 100, 1000, 300, step=100, key="spy_size")
    rng = np.random.default_rng(42)
    sample_u = rng.choice(n_users, size=min(spy_size, n_users), replace=False)
    sample_i = rng.choice(n_items, size=min(spy_size, n_items), replace=False)
    sample_u.sort()
    sample_i.sort()

    item_set = set(sample_i.tolist())
    u_idx = {u: i for i, u in enumerate(sample_u)}
    i_idx = {it: i for i, it in enumerate(sample_i)}

    rows, cols = [], []
    for u in sample_u:
        u_int = int(u)
        if u_int in train_dict:
            for it in train_dict[u_int]:
                if it in item_set:
                    rows.append(u_idx[u_int])
                    cols.append(i_idx[it])

    fig_spy = go.Figure()
    fig_spy.add_trace(
        go.Scatter(
            x=cols,
            y=rows,
            mode="markers",
            marker=dict(size=2, color="navy"),
        )
    )
    fig_spy.update_layout(
        height=500,
        xaxis_title="Items (sampled)",
        yaxis_title="Users (sampled)",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_spy, use_container_width=True)

    # --- 2-hop subgraph ---
    st.subheader("2-Hop Neighborhood Subgraph")
    seed_user = st.number_input(
        "Seed user ID", min_value=0, max_value=n_users - 1, value=0, key="seed_user"
    )
    if seed_user not in train_dict or len(train_dict[seed_user]) == 0:
        st.warning(f"User {seed_user} has no training interactions. Try another user.")
        return

    # 1-hop: user -> items
    hop1_items = train_dict[seed_user]
    # 2-hop: items -> other users (limit for readability)
    hop2_users: set[int] = set()
    for it in hop1_items:
        for u, items in train_dict.items():
            if it in items and u != seed_user:
                hop2_users.add(u)
            if len(hop2_users) >= 50:
                break
        if len(hop2_users) >= 50:
            break

    # Build network positions
    node_x, node_y, node_text, node_color = [], [], [], []
    edge_x, edge_y = [], []

    # Seed user at center
    node_x.append(0.0)
    node_y.append(0.0)
    node_text.append(f"User {seed_user} (seed)")
    node_color.append("red")

    # Items in a ring
    for idx, it in enumerate(hop1_items):
        angle = 2 * np.pi * idx / len(hop1_items)
        ix, iy = np.cos(angle) * 2, np.sin(angle) * 2
        node_x.append(ix)
        node_y.append(iy)
        node_text.append(f"Item {it}")
        node_color.append("steelblue")
        edge_x.extend([0.0, ix, None])
        edge_y.extend([0.0, iy, None])

    # 2-hop users in outer ring
    hop2_list = list(hop2_users)[:30]
    for idx, u in enumerate(hop2_list):
        angle = 2 * np.pi * idx / max(len(hop2_list), 1)
        ux, uy = np.cos(angle) * 4, np.sin(angle) * 4
        node_x.append(ux)
        node_y.append(uy)
        node_text.append(f"User {u}")
        node_color.append("orange")
        # Connect to shared items
        for jdx, it in enumerate(hop1_items):
            if it in train_dict.get(u, []):
                item_angle = 2 * np.pi * jdx / len(hop1_items)
                ix2, iy2 = np.cos(item_angle) * 2, np.sin(item_angle) * 2
                edge_x.extend([ux, ix2, None])
                edge_y.extend([uy, iy2, None])

    fig_net = go.Figure()
    fig_net.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#ccc"),
            hoverinfo="none",
        )
    )
    fig_net.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=10, color=node_color, line=dict(width=1, color="black")),
        )
    )
    fig_net.update_layout(
        height=600,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig_net, use_container_width=True)
