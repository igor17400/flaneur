---
name: ablation
description: Run ablation studies using Hydra multirun. Sweeps one parameter at a time, auto-tags each run in W&B, and compares results to find the best value.
allowed-tools: Read, Edit, Write, Bash, Glob, Grep
---

# Ablation Study

Run a controlled ablation study: sweep one parameter across multiple values using Hydra multirun, then compare results in W&B.

## Principles

- **One parameter per sweep** — isolate the effect of a single variable
- **Use Hydra `--multirun`** — launches multiple runs sequentially from one command
- **Auto-tagged** — each run gets W&B notes/tags from the config comment (already wired in main.py)
- **Compare after** — fetch results from W&B and present a comparison table

## Step 1: Choose What to Ablate

Ask the user which parameter to sweep, or pick from the latest /analyze-run recommendations. Common ablations:

| Parameter | Config key | Typical values |
|-----------|-----------|----------------|
| Embedding dim | `model.embed_dim` | 32,64,128,256 |
| GCN layers | `model.n_layers` | 2,3,4 |
| Learning rate | `train.lr` | 5e-4,1e-3,2e-3 |
| L2 regularization | `train.reg_weight` | 1e-5,1e-4,1e-3 |
| Batch size | `train.batch_size` | 1024,2048,4096 |
| Negative samples | `train.n_negatives` | 1,3,5 |
| Embedding dropout | `model.embed_dropout` | 0.0,0.1,0.2 |

## Step 2: Update Config Comment

Before running, update the config comment to describe the ablation:

```yaml
# Ablation: sweeping {param} over [{values}].
# Base config: {describe the fixed parameters}.
```

This ensures each run gets the right W&B notes via `parse_config_notes()`.

## Step 3: Run the Sweep

Use Hydra multirun syntax. The `-m` flag tells Hydra to run all combinations:

```bash
uv run python src/main.py -m experiment=lgcn_gowalla_full {config.key}={v1},{v2},{v3}
```

### Examples

```bash
# Sweep n_negatives
uv run python src/main.py -m experiment=lgcn_gowalla_full train.n_negatives=1,3,5

# Sweep reg_weight
uv run python src/main.py -m experiment=lgcn_gowalla_full train.reg_weight=1e-5,1e-4,1e-3

# Sweep embed_dim
uv run python src/main.py -m experiment=lgcn_gowalla_full model.embed_dim=64,128

# Sweep n_layers
uv run python src/main.py -m experiment=lgcn_gowalla_full model.n_layers=2,3,4
```

**Important**: Hydra multirun runs sequentially by default. For N values, total time = N * single_run_time. Keep sweep values to 3-4 for hackathon speed.

## Step 4: Analyze Results

After the sweep completes, fetch all runs from W&B and compare:

```python
import wandb
api = wandb.Api()
runs = api.runs("igorlima1740/flaneur", order="-created_at")

# Filter to recent sweep runs and compare
for r in runs[:N]:  # N = number of sweep values
    cfg = dict(r.config)
    s = dict(r.summary)
    print(f"{param}={cfg[param_key]} → Recall={s.get('best_val_recall@20'):.4f}")
```

Present a clear comparison table:

```
| {param} | Recall@20 | NDCG@20 | BPR Loss | Winner? |
|---------|-----------|---------|----------|---------|
| value1  | ...       | ...     | ...      |         |
| value2  | ...       | ...     | ...      | ✓       |
| value3  | ...       | ...     | ...      |         |
```

## Step 5: Apply the Winner

After identifying the best value:
1. Update `configs/experiment/lgcn_gowalla_full.yaml` with the winning value
2. Update the config comment to document the finding
3. Report the result

## Full Workflow Example

User says: "ablate n_negatives"

1. Read current config to understand baseline
2. Update config comment: `# Ablation: n_negatives over [1,3,5]. Base: 128-dim/4-layer.`
3. Run: `uv run python src/main.py -m experiment=lgcn_gowalla_full train.n_negatives=1,3,5`
4. Fetch results from W&B, build comparison table
5. Apply winner to config
