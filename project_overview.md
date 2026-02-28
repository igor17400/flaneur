# Flaneur - Self-Improving LightGCN on Gowalla

## Hackathon Context

This project is for the **W&B Mini Challenge: Best Self-Improvement Workflow** at the Mistral Worldwide Hackathon 2026 (Feb 28 - Mar 1).

**Prize:** $500 CoreWeave Inference Credits + Mac Mini

**Goal:** Demonstrate that a coding agent (Claude Code) can automatically evaluate, optimize, and improve an AI application using W&B MCP tools.

### Judging Criteria (Mini Challenge)

| Criteria | Description | Required? |
|---|---|---|
| Proven Improvement | Measurable metric increase via agent-driven automation | Required |
| Generated Skills Submitted | Skills/prompts/configs the agent generated | Required |
| Creativity | How creative and novel is the automation approach | Judged |
| Completeness | End-to-end workflow: eval -> analysis -> improvement | Judged |

### What Judges Want to See

1. **Automated evals creation** - agent uses W&B MCP tools to create and run evaluations
2. **Optimization loop** - agent inspects results and iterates to improve performance
3. **Smart delegation** - creative use of W&B tools to let the model drive the improvement cycle

---

## Project: LightGCN on Gowalla

### What It Is

A JAX-based implementation of **LightGCN** (Light Graph Convolutional Network) for collaborative filtering on the **Gowalla** check-in dataset. LightGCN simplifies GCN by removing feature transformations and nonlinear activations, keeping only neighborhood aggregation on a user-item bipartite graph.

### Key Metrics

- **Recall@20** - primary evaluation metric
- **NDCG@20** - secondary evaluation metric
- **BPR Loss** - training loss (Bayesian Personalized Ranking)

### LightGCN Reference Benchmarks (Gowalla)

From the original LightGCN paper:
- Recall@20 ~ 0.183
- NDCG@20 ~ 0.154

---

## Project Structure

```
flaneur/
├── configs/
│   ├── config.yaml                # Root Hydra config (composes model + data + experiment)
│   ├── data/gowalla.yaml          # Dataset config
│   ├── model/lightgcn.yaml        # Model: embed_dim=64, n_layers=3
│   └── experiment/
│       ├── default.yaml           # Full run: 1000 epochs, W&B enabled
│       ├── lgcn_gowalla_full.yaml # Named full run
│       └── lgcn_gowalla_test.yaml # 5-epoch smoke test, W&B disabled
│
├── data/gowalla/
│   ├── train.txt                  # 29,858 users of training interactions
│   └── test.txt                   # 29,858 users of test interactions
│
├── src/
│   ├── main.py          # Entry point: Hydra + W&B init + train()
│   ├── model.py         # LightGCN forward pass (JAX) + BPR loss
│   ├── train.py         # Training loop: JIT, mini-batching, eval, W&B logging
│   ├── data.py          # Data loading, adjacency matrix, negative sampling
│   └── evaluate.py      # Recall@K and NDCG@K (full-ranking evaluation)
│
└── vis_spatial/         # Streamlit dataset explorer (4 tabs)
    ├── app.py
    ├── about.md
    ├── utils.py
    └── (tab modules)
```

### Running Training

```bash
cd flaneur
# Full run (1000 epochs, W&B enabled)
uv run python src/main.py experiment=default

# Smoke test (5 epochs, no W&B)
uv run python src/main.py experiment=lgcn_gowalla_test

# Override hyperparams via Hydra CLI
uv run python src/main.py experiment=default train.lr=0.005 train.batch_size=4096 model.n_layers=4
```

### Current Default Hyperparameters

| Parameter | Value | Config Key |
|---|---|---|
| Learning rate | 1e-3 | `train.lr` |
| L2 regularization | 1e-4 | `train.reg_weight` |
| Batch size | 2048 | `train.batch_size` |
| Epochs | 1000 | `train.epochs` |
| Embedding dim | 64 | `model.embed_dim` |
| GCN layers | 3 | `model.n_layers` |
| Top-K for eval | 20 | `train.topk` |
| Eval frequency | every 10 epochs | `train.eval_every` |
| Seed | 2020 | `train.seed` |

---

## W&B Integration (Already Wired)

- **W&B entity:** `igorlima1740`
- **W&B project:** `flaneur`
- **Metrics logged per epoch:** `train/bpr_loss`, `train/epoch_time_sec`
- **Metrics logged every eval_every epochs:** `eval/recall@20`, `eval/ndcg@20`
- **Summary metrics:** `best_recall@20`, `best_ndcg@20`
- **Config:** Full Hydra config passed to `wandb.init(config=...)`

### W&B MCP Tools Available

The agent has access to these W&B MCP tools:
- `query_wandb_tool` - Query runs, metrics, experiments via GraphQL
- `query_weave_traces_tool` - Query Weave traces (LLM observability)
- `count_weave_traces_tool` - Count Weave traces
- `create_wandb_report_tool` - Create W&B Reports
- `query_wandb_entity_projects` - List projects
- `query_wandb_support_bot` - W&B documentation help

---

## Self-Improvement Workflow

The agent should follow this loop:

### Phase 1: Baseline Run
1. Run LightGCN with default hyperparameters
2. Log everything to W&B (`flaneur` project)
3. Record baseline Recall@20 and NDCG@20

### Phase 2: Analyze Results via W&B MCP
1. Use `query_wandb_tool` to fetch run metrics (loss curves, eval metrics)
2. Analyze: Is the model converging? Overfitting? Underfitting?
3. Identify bottlenecks (e.g., loss plateauing, poor recall growth)

### Phase 3: Propose & Apply Improvements
Based on analysis, adjust hyperparameters. Key levers:

| Lever | What to Try | Why |
|---|---|---|
| `train.lr` | 5e-4, 1e-3, 5e-3 | Learning rate is the most impactful hyperparameter |
| `model.embed_dim` | 32, 64, 128 | Larger embeddings = more capacity but risk overfitting |
| `model.n_layers` | 2, 3, 4 | More layers = more neighborhood info but over-smoothing risk |
| `train.reg_weight` | 1e-5, 1e-4, 1e-3 | Controls overfitting via L2 regularization |
| `train.batch_size` | 1024, 2048, 4096 | Affects gradient noise and convergence speed |

Also consider code-level improvements:
- Learning rate scheduling (e.g., cosine decay)
- Better negative sampling strategies
- Layer-wise attention weights instead of uniform averaging
- Dropout on embeddings
- Early stopping based on eval metrics

### Phase 4: Retrain & Compare
1. Run improved config, log to W&B
2. Use W&B MCP to compare runs
3. Repeat Phase 2-4 until metrics improve

### Phase 5: Report
1. Use `create_wandb_report_tool` to generate a W&B Report
2. Include: training curves, metric comparisons, what the agent changed and why

---

## Dependencies

Managed with `uv` (Python 3.10):
- `jax`, `jaxlib` - ML framework
- `optax` - Optimizers
- `hydra-core`, `omegaconf` - Config management
- `wandb` - Experiment tracking
- `scipy`, `numpy` - Sparse matrix ops
- `streamlit`, `plotly` - Visualization
- `tqdm` - Progress bars

---

## Key Notes for the Agent

1. **Always use `uv run`** to execute Python in the project's virtual environment
2. **Working directory for training:** `flaneur/` (Hydra expects configs/ to be relative)
3. **Hydra overrides** via CLI are the fastest way to change hyperparameters
4. **W&B must be enabled** (`experiment.wandb.enabled=true`) for tracking
5. **Eval happens every `eval_every` epochs** - set this appropriately for run length
6. **The dataset auto-downloads** on first run if `data/gowalla/` is empty
7. **JAX runs on CPU by default** on Mac - training may be slow for 1000 epochs; consider shorter runs (100-200 epochs) for iteration speed
8. **Check W&B runs** via MCP after each training to analyze results before next iteration
