# Flaneur + Derive — Project State

> Last updated: 2026-02-28 (Mistral Worldwide Hackathon, Day 1)

## Hackathon Context

- **Event:** Mistral Worldwide Hackathon 2026 (Feb 28 – Mar 1)
- **Track:** Mistral AI (Agents) — build anything with the Mistral API
- **Mini Challenge:** W&B Best Self-Improvement Workflow ($500 CoreWeave + Mac Mini)
- **Submission platform:** hackiterate.com
- **Judging:** Technicality, Creativity, Usefulness, Demo, Track alignment

We are targeting **both** the main Mistral track and the W&B mini challenge.

---

## Two Projects, One Repo

### 1. Flaneur (LightGCN training + self-improvement loop)

**Purpose:** Demonstrates an automated eval → analyze → improve loop using Claude Code + W&B MCP tools. This is the **W&B mini challenge** submission.

**What it does:**
- Trains a LightGCN (Light Graph Convolutional Network) model for collaborative filtering on the Gowalla check-in dataset
- Logs metrics to W&B (`igorlima1740/flaneur`)
- Agent uses W&B MCP tools to analyze runs, diagnose issues, apply improvements, and retrain
- Final evaluation via W&B Weave with Mistral LLM reflection

**Key metrics:** Recall@20 (primary), NDCG@20 (secondary)
**Reference benchmarks:** Recall@20 ~ 0.183, NDCG@20 ~ 0.154 (from LightGCN paper)

**Status:** Core training pipeline is complete. Self-improvement skills (`/analyze-run`, `/improve-model`, `/eval-report`) are defined. Training runs have NOT been started yet.

### 2. Derive (interactive map visualization)

**Purpose:** Browser-based visualization of LightGCN recommendations on a real world map. This is the **main hackathon** submission — it makes RecSys results interpretable and demo-able.

**What it does:**
- Maps every LightGCN item ID back to real lat/lon coordinates via SNAP Gowalla data
- Shows any user's check-in history (train set) and ground truth (test set) on an interactive dark-themed map
- Numbered markers, timeline panel, hover-to-highlight, click-to-fly, animated playback
- Will integrate Mistral API for per-user LLM diagnostics (not built yet)

**Status:** Working prototype. The map, timeline, search, animation, and user lookup all work. See "What's Next" for remaining work.

---

## Repository Structure

```
flaneur_main_hackatho/
│
├── CLAUDE.md                   # Agent instructions for the self-improvement loop
├── PROJECT_STATE.md            # ← THIS FILE (handoff doc for next agent)
├── project_overview.md         # Detailed technical docs (model, metrics, configs)
├── pyproject.toml              # Dependencies (uv, Python 3.10+)
├── README.md                   # (empty)
│
├── configs/                    # Hydra config management
│   ├── config.yaml             # Root config (composes model + data + experiment)
│   ├── model/lightgcn.yaml     # embed_dim=64, n_layers=3
│   ├── data/gowalla.yaml       # Dataset path
│   └── experiment/
│       ├── default.yaml        # Full run: 1000 epochs, W&B enabled
│       ├── lgcn_gowalla_test.yaml   # 2-epoch smoke test
│       └── lgcn_gowalla_full.yaml   # Full + embed_dropout + early stopping
│
├── src/                        # LightGCN training code (JAX)
│   ├── main.py                 # Entry point (Hydra + W&B init)
│   ├── model.py                # LightGCN forward pass + BPR loss
│   ├── train.py                # Training loop (cosine LR, Adam, eval, W&B logging)
│   ├── data.py                 # Data loading, adjacency matrix, negative sampling
│   ├── evaluate.py             # Recall@K, NDCG@K evaluation
│   └── evaluate_weave.py       # Weave-traced eval + Mistral reflection
│
├── prompts/
│   └── reflection.txt          # Mistral prompt for analyzing eval results
│
├── data/                       # Auto-downloaded, gitignored
│   ├── gowalla/
│   │   ├── train.txt           # LightGCN format: userID item1 item2 ...
│   │   ├── test.txt
│   │   ├── item_list.txt       # CRITICAL: org_id → remap_id mapping for items
│   │   └── user_list.txt       # CRITICAL: org_id → remap_id mapping for users
│   └── gowalla_raw/
│       ├── loc-gowalla_totalCheckins.txt      # Original SNAP data (376MB)
│       ├── loc-gowalla_totalCheckins.txt.gz
│       └── demo_users.json     # (legacy, can delete)
│
├── derive/                     # ★ Map visualization app
│   ├── server.py               # HTTP server + JSON API
│   ├── lib/
│   │   ├── __init__.py
│   │   └── gowalla.py          # Data loading: LightGCN ↔ SNAP ID bridge + geo lookup
│   └── static/
│       ├── index.html           # Main page (HTML structure only)
│       ├── css/style.css        # All styles (dark theme, sidebar, timeline)
│       └── js/
│           ├── map.js           # MapLibre + deck.gl (layers, fit, fly, tooltips)
│           ├── timeline.js      # Sidebar timeline, stats, recent chips, hover
│           ├── animation.js     # Playback animation controller
│           └── app.js           # Main controller (wires search, map, timeline)
│
├── vis_spatial/                 # Streamlit dataset explorer (4 tabs, statistics)
│   ├── app.py
│   ├── about.md
│   ├── utils.py
│   └── (tab modules)
│
└── .claude/skills/             # Claude Code custom skills
    ├── analyze-run/SKILL.md
    ├── improve-model/SKILL.md
    └── eval-report/SKILL.md
```

---

## Data Pipeline (IMPORTANT)

The LightGCN model uses remapped integer IDs (0-indexed, contiguous). The original SNAP Gowalla data has sparse IDs but includes lat/lon coordinates. The bridge works like this:

```
LightGCN item 42
    → item_list.txt says remap_id 42 = org_id 23360
    → SNAP checkins say location 23360 is at (39.04, -94.59)
    → Pin on map: Kansas City, MO
```

**Coverage:** 40,980 out of 40,981 items (99.99%) successfully map to coordinates.

**Key files for the mapping:**
- `data/gowalla/item_list.txt` — `org_id remap_id` (header, then one pair per line)
- `data/gowalla/user_list.txt` — same format for users
- `data/gowalla_raw/loc-gowalla_totalCheckins.txt` — `user\ttimestamp\tlat\tlon\tlocation_id`

**Loading:** `derive/lib/gowalla.py` loads everything into memory (~2GB RAM, takes ~15 seconds).

---

## Running Things

### Derive visualization
```bash
cd flaneur_main_hackatho

# Kill any old server on port 8765
lsof -ti:8765 | xargs kill -9 2>/dev/null

# Start server (loads data in ~15s, then serves on localhost:8765)
python3 derive/server.py
```

### LightGCN training
```bash
cd flaneur_main_hackatho

# Smoke test (2 epochs)
uv run python src/main.py experiment=lgcn_gowalla_test

# Full training (1000 epochs, logs to W&B)
uv run python src/main.py experiment=default

# Custom hyperparams
uv run python src/main.py experiment=default train.lr=0.005 model.embed_dim=128
```

### Weave evaluation
```bash
uv run python src/evaluate_weave.py --run default --n_users 200
```

---

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| ML model | JAX + Optax | LightGCN, BPR loss, cosine LR decay |
| Config | Hydra + OmegaConf | YAML configs, CLI overrides |
| Experiment tracking | W&B Models | Entity: `igorlima1740`, project: `flaneur` |
| LLM evaluation | W&B Weave + Mistral | Traced evaluation with LLM reflection |
| Map rendering | MapLibre GL JS | Open-source, no API key, GPU-accelerated |
| Data viz layers | deck.gl | Scatter, path, arc, text layers via MapboxOverlay |
| Map tiles | CartoDB dark | Free, no API key |
| Backend | Python http.server | Zero dependencies, stdlib only |
| Frontend | Vanilla JS | No framework, no build step |
| Package management | uv | Python 3.10+ |

---

## What's Done

- [x] LightGCN training pipeline (JAX) with W&B logging
- [x] Hydra config system with experiment variants
- [x] Weave evaluation + Mistral reflection
- [x] Claude Code skills for self-improvement loop
- [x] SNAP Gowalla data downloaded and ID mapping verified (100% coverage)
- [x] Derive: map visualization with dark theme, deck.gl layers
- [x] Derive: numbered markers, path line (not arc spaghetti)
- [x] Derive: timeline panel with hover-to-highlight and click-to-fly
- [x] Derive: user search by ID, random user, recent user chips
- [x] Derive: animation playback with progress bar
- [x] Derive: modular code structure (map.js, timeline.js, animation.js, app.js)
- [x] Platform-conditional deps (JAX CPU on Mac, CUDA on Linux)

## What's Next

### High Priority (hackathon must-haves)
- [ ] **Run baseline training** — execute the self-improvement loop, get real metrics
- [ ] **Integrate model predictions into Derive** — currently showing ground truth as "predictions"; after training, load checkpoint embeddings and run actual inference to get top-K recommendations
- [ ] **Mistral LLM diagnostics** — add a "Diagnose" button that sends user history + predictions to Mistral API and streams the interpretation
- [ ] **W&B Report** — generate final report via `create_wandb_report_tool`

### Medium Priority (polish for demo)
- [ ] **Predictions vs ground truth view** — show model's predictions in a third color, overlay with actual ground truth to visualize hits/misses
- [ ] **Aggregate insights** — Mistral-powered analysis of user clusters/patterns
- [ ] **Better animation** — camera following the user's journey, smooth transitions
- [ ] **Mobile/responsive layout** — sidebar collapse for smaller screens

### Nice to Have
- [ ] Reverse geocoding (show place names, not just coords)
- [ ] User similarity — "users like this one" based on embedding distance
- [ ] Embedding space view — UMAP projection of user/item embeddings

---

## Environment Notes

- **Mac (dev):** JAX runs on CPU (Metal). Training is slow — use short runs (100-200 epochs) for iteration.
- **Linux (prod/GPU):** JAX with CUDA 12. Full 1000-epoch runs feasible.
- **pyproject.toml** has platform-conditional deps: `jax-cuda12-plugin` only installs on Linux.
- **Data files are gitignored.** They auto-download on first run (`src/data.py` handles LightGCN data, SNAP data must be downloaded separately for Derive).

## W&B Integration

- **Entity:** `igorlima1740`
- **Project:** `flaneur`
- **Metrics logged:** `train/bpr_loss`, `train/epoch_time_sec`, `val/recall@20`, `val/ndcg@20`
- **Summary:** `best_recall@20`, `best_ndcg@20`
- **MCP tools available:** `query_wandb_tool`, `query_weave_traces_tool`, `count_weave_traces_tool`, `create_wandb_report_tool`, `query_wandb_entity_projects`, `query_wandb_support_bot`
