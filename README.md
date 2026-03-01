# Flaneur

[![Demo Video](https://img.shields.io/badge/Demo-YouTube-red?logo=youtube)](https://youtu.be/s6rvHhWhWIA)

A self-improving LightGCN recommender system for location check-in prediction on the Gowalla dataset. Built with JAX, Mistral, W&B Weave, and Claude Code.

Flaneur combines a JAX-based LightGCN model with an agentic self-improvement loop: train, analyze metrics via W&B MCP, apply fixes, evaluate with Weave, and repeat — all orchestrated through Claude Code skills.

The project also ships **Derive**, a browser-based visualization app with an interactive map and a Mistral-powered chat agent that explains user behavior, compares movement patterns, and connects behavioral insights to model performance.

## Architecture

```
                  ┌──────────────┐
            ┌────>│  /analyze-run │ Query W&B MCP for metrics
            │     └──────┬───────┘
            │            v
┌───────────┴──┐  ┌──────────────┐
│   Training   │  │ /improve-model│ Edit config/code based on analysis
│  (JAX + BPR) │  └──────┬───────┘
└───────┬──────┘         │
        │           uv run python src/main.py
        │                │
        v                v
┌──────────────┐  ┌──────────────┐
│  Checkpoints │  │  /eval-report │ Weave evaluation + W&B Report
└──────────────┘  └──────────────┘
        │
        v
┌──────────────────────────────────┐
│  Derive (localhost:8765)         │
│  Interactive map + Mistral chat  │
│  Behavioral analysis tools       │
└──────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
uv sync

# Train with default config
uv run python src/main.py experiment=lgcn_gowalla_full

# Override hyperparameters
uv run python src/main.py experiment=lgcn_gowalla_full train.lr=0.005 model.embed_dim=128

# Run inference (generates predictions/*.json)
uv run python src/infer.py --checkpoint checkpoints/<name>

# Launch the Derive visualization app
uv run python derive/server.py
# Open http://localhost:8765

# Run Weave evaluation on a checkpoint
uv run python src/evaluate_weave.py --run lgcn_gowalla_full --n_users 200
```

## Project Structure

```
flaneur/
├── src/                    # Core ML code
│   ├── main.py             # Hydra entry point, training orchestration
│   ├── model.py            # LightGCN forward pass, BPR loss (JAX)
│   ├── train.py            # Training loop, cosine LR, early stopping
│   ├── data.py             # Gowalla loader, adjacency matrix, negative sampling
│   ├── evaluate.py         # Recall@K, NDCG@K batched evaluation
│   ├── infer.py            # Top-K inference, prediction export
│   └── evaluate_weave.py   # Weave-integrated eval with Mistral reflection
│
├── derive/                 # Visualization server
│   ├── server.py           # HTTP server + Mistral agentic chat (SSE)
│   ├── lib/
│   │   ├── gowalla.py      # Data bridge: LightGCN IDs <-> SNAP geo coordinates
│   │   └── agent_tools.py  # Tool definitions: analyze_behavior, compare_users, etc.
│   └── static/             # Frontend (HTML, JS, CSS, Leaflet map)
│
├── configs/                # Hydra YAML configs
│   ├── experiment/         # Training run configs (lgcn_gowalla_full.yaml)
│   ├── model/              # Model architecture (lightgcn.yaml)
│   └── data/               # Dataset config (gowalla.yaml)
│
├── prompts/                # System prompts
│   ├── agent_system.txt    # Derive chat agent (behavioral data scientist)
│   ├── reflection.txt      # Mistral evaluation reflection
│   └── explain_user.txt    # User behavior explanation
│
├── .claude/skills/         # Claude Code automation skills
├── terraform/              # AWS EC2 deployment
├── checkpoints/            # Saved model embeddings
├── predictions/            # Top-K predictions with scores (JSON)
└── vis_spatial/            # Streamlit dataset explorer
```

## Self-Improvement Loop

The project uses seven Claude Code skills in a loop:

| Skill                | Purpose                                         |
| -------------------- | ----------------------------------------------- |
| `/analyze-run`       | Query W&B for training metrics, diagnose issues |
| `/improve-model`     | Apply config/code changes based on analysis     |
| `/apply-fix`         | Implement a single focused improvement          |
| `/ablation`          | Sweep one parameter with Hydra multirun         |
| `/weave-eval`        | Quick Weave evaluation on test set              |
| `/eval-report`       | Full Weave eval + W&B Report for the journey    |
| `/project-changelog` | Generate a W&B Report tracking all changes      |

Typical workflow:

```
/analyze-run  ->  /improve-model  ->  train  ->  /analyze-run  ->  ...  ->  /eval-report
```

## Model

**LightGCN** (He et al., 2020) -- a graph convolutional network for collaborative filtering that learns user and item embeddings by propagating them on the user-item interaction graph.

- **Loss**: Bayesian Personalized Ranking (BPR) with L2 regularization
- **Adjacency**: Symmetric normalized (D^{-1/2} A D^{-1/2})
- **LR schedule**: Cosine decay to 1% of initial LR
- **Negative sampling**: Configurable N negatives per positive
- **Evaluation**: Recall@20, NDCG@20 (paper benchmarks: 0.183, 0.154)

Key hyperparameters (in `configs/experiment/lgcn_gowalla_full.yaml`):

| Parameter           | Default | Description                   |
| ------------------- | ------- | ----------------------------- |
| `model.embed_dim`   | 128     | Embedding dimension           |
| `model.n_layers`    | 4       | GCN propagation layers        |
| `train.lr`          | 1e-3    | Learning rate                 |
| `train.reg_weight`  | 1e-5    | L2 regularization             |
| `train.batch_size`  | 2048    | Mini-batch size               |
| `train.n_negatives` | 3       | Negative samples per positive |
| `train.epochs`      | 100     | Training epochs               |

## Dataset

**Gowalla** -- a location-based social network (2009-2012) from the SNAP dataset:

- **29,858** users
- **~18,000** venues (locations with lat/lon)
- **~1.3M** training interactions
- Auto-downloads on first run

## Derive -- Visualization App

An interactive browser app for exploring user check-in patterns and model predictions.

**Features:**

- Interactive Leaflet map with history, ground truth, and prediction layers
- Mistral-powered chat agent with behavioral analysis tools
- User comparison, behavioral profiling, movement classification
- Full-screen reports with charts and PDF export
- Leaderboard and global heatmap views
- Weave-traced conversations (each chat turn links to W&B)

**Chat Agent Tools:**

- `analyze_behavior(uid)` -- Movement style, revisit patterns, geographic clustering, prediction accuracy breakdown
- `compare_users(uid_a, uid_b)` -- Side-by-side behavioral comparison with similarity score (0-100)
- `lookup_user`, `get_user_detail`, `find_users`, `get_model_stats` -- Data retrieval
- `navigate_map`, `show_widget`, `generate_report` -- Visualization

## W&B Integration

- **Entity**: `igorlima1740` | **Project**: `flaneur`
- **Training**: Logs BPR loss, epoch time, val recall/NDCG per epoch
- **Weave**: Traces Derive chat conversations, evaluation runs with Mistral reflection
- **Reports**: Auto-generated summaries of self-improvement iterations

## Deployment

```bash
# AWS deployment via Terraform
cd terraform
terraform init
terraform apply
```

Provisions an EC2 instance with the Derive server accessible on port 80.

## Tech Stack

| Layer      | Technology                                |
| ---------- | ----------------------------------------- |
| Model      | JAX, Optax, SciPy (sparse)                |
| Config     | Hydra, OmegaConf                          |
| Tracking   | W&B, Weave                                |
| Chat agent | Mistral (function calling, SSE streaming) |
| Automation | Claude Code skills                        |
| Frontend   | Leaflet, Chart.js, vanilla JS             |
| Server     | Python stdlib (http.server, threading)    |
| Deploy     | Terraform, AWS EC2                        |

## License

Hackathon project -- Mistral Worldwide Hackathon.
