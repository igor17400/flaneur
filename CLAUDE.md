# Flaneur — LightGCN Self-Improvement Workflow

## Project
JAX-based LightGCN for collaborative filtering on the Gowalla dataset.
W&B entity: `igorlima1740`, project: `flaneur`.

## Self-Improvement Loop

This project uses Claude Code skills in two phases:

### Training Loop
1. **`/analyze-run`** — Query W&B MCP for training metrics, diagnose issues
2. **`/improve-model`** — Apply config/code changes based on analysis

### Inference & Evaluation
3. **`/weave-eval`** — A/B test best models on pre-computed predictions
4. **`/eval-report`** — Generate W&B Report with full evaluation results

### Workflow
```
Training:   /analyze-run → /improve-model → train → /analyze-run → ...
Inference:  infer.py → /weave-eval → /eval-report
```

## Key Commands

```bash
# Train with the experiment config
uv run python src/main.py experiment=lgcn_gowalla_full

# Override hyperparams via CLI
uv run python src/main.py experiment=lgcn_gowalla_full train.lr=0.005 model.embed_dim=128

# Run Weave evaluation on a checkpoint
uv run python src/evaluate_weave.py --run lgcn_gowalla_full --n_users 200

# List available checkpoints
uv run python src/evaluate_weave.py --list
```

## Conventions
- Use `uv run` for all Python execution
- Edit `configs/experiment/lgcn_gowalla_full.yaml` in-place (do NOT create v2, v3, etc.)
- W&B and git history track the evolution across runs
- Always include a comment at the top of the config explaining what changed
- Checkpoints save to `checkpoints/{run_name}/`
- Prompts for Mistral reflection live in `prompts/`
