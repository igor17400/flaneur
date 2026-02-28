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

    # --- Small readable subgraph ---
    st.subheader("Bipartite Structure (small subgraph)")
    st.caption(
        "A tiny slice: pick a few items from the seed user, then show a few other "
        "users who also checked in at those items. This makes the user ↔ item "
        "bipartite structure easy to read."
    )

    col_a, col_b = st.columns(2)
    max_items = col_a.slider("Items to show", 2, 10, 4, key="mini_items")
    max_neighbors = col_b.slider(
        "Neighbor users per item", 1, 5, 2, key="mini_neighbors"
    )

    # Build an inverted index: item -> list of users (excluding seed)
    rng_mini = np.random.default_rng(seed_user)
    seed_items = list(train_dict[seed_user])
    rng_mini.shuffle(seed_items)
    picked_items = seed_items[:max_items]

    item_to_users: dict[int, list[int]] = {it: [] for it in picked_items}
    for u, items in train_dict.items():
        if u == seed_user:
            continue
        u_items = set(items)
        for it in picked_items:
            if it in u_items and len(item_to_users[it]) < max_neighbors:
                item_to_users[it].append(u)

    # Layout: 3 columns — seed user (left), items (center), neighbor users (right)
    sx, sy = [], []  # node positions
    st_text, st_color, st_size = [], [], []
    ex, ey = [], []  # edges

    # Seed user
    sx.append(0.0)
    sy.append(0.0)
    st_text.append(f"U{seed_user}")
    st_color.append("#e74c3c")
    st_size.append(18)

    # Items column (x=2), evenly spaced vertically
    item_positions: dict[int, tuple[float, float]] = {}
    for idx, it in enumerate(picked_items):
        iy = (idx - (len(picked_items) - 1) / 2) * 1.2
        item_positions[it] = (2.0, iy)
        sx.append(2.0)
        sy.append(iy)
        st_text.append(f"I{it}")
        st_color.append("#3498db")
        st_size.append(14)
        # Edge: seed -> item
        ex.extend([0.0, 2.0, None])
        ey.extend([0.0, iy, None])

    # Neighbor users column (x=4)
    neighbor_y_offset = 0.0
    placed_neighbors: dict[int, tuple[float, float]] = {}
    for it in picked_items:
        ix, iy = item_positions[it]
        for j, u in enumerate(item_to_users[it]):
            if u not in placed_neighbors:
                ny = iy + (j - (len(item_to_users[it]) - 1) / 2) * 0.6
                placed_neighbors[u] = (4.0, ny)
                sx.append(4.0)
                sy.append(ny)
                st_text.append(f"U{u}")
                st_color.append("#f39c12")
                st_size.append(14)
            ux, uy = placed_neighbors[u]
            # Edge: item -> neighbor user
            ex.extend([ix, ux, None])
            ey.extend([iy, uy, None])

    fig_mini = go.Figure()
    fig_mini.add_trace(
        go.Scatter(
            x=ex,
            y=ey,
            mode="lines",
            line=dict(width=1.5, color="#bbb"),
            hoverinfo="none",
        )
    )
    fig_mini.add_trace(
        go.Scatter(
            x=sx,
            y=sy,
            mode="markers+text",
            text=st_text,
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=st_size, color=st_color, line=dict(width=1, color="#333")),
            hoverinfo="text",
        )
    )
    fig_mini.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"
        ),
    )
    st.plotly_chart(fig_mini, use_container_width=True)

    st.markdown(
        "**Legend:** "
        ":red[seed user] · "
        ":blue[items] · "
        ":orange[neighbor users] — "
        "Edges show check-ins. Two users are *similar* if they share item neighbors."
    )
