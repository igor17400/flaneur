# Flaneur — LightGCN Self-Improvement Workflow

## Project
JAX-based LightGCN for collaborative filtering on the Gowalla dataset.
W&B entity: `igorlima1740`, project: `flaneur`.

## Self-Improvement Loop

This project uses three Claude Code skills in a loop:

1. **`/analyze-run`** — Query W&B MCP for training metrics, diagnose issues
2. **`/improve-model`** — Apply config/code changes based on analysis
3. **`/eval-report`** — Run Weave test evaluation and generate W&B Report

### Workflow
```
/analyze-run → /improve-model → train → /analyze-run → ... → /eval-report
```

## Key Commands

```bash
# Train with a specific experiment config
uv run python src/main.py experiment=lgcn_gowalla_v2

# Override hyperparams via CLI
uv run python src/main.py experiment=default train.lr=0.005 model.embed_dim=128

# Run Weave evaluation on a checkpoint
uv run python src/evaluate_weave.py --run lgcn_gowalla_v2 --n_users 200

# List available checkpoints
uv run python src/evaluate_weave.py --list
```

## Conventions
- Use `uv run` for all Python execution
- Experiment configs go in `configs/experiment/`
- Name configs as `lgcn_gowalla_v{N}.yaml` with increment per iteration
- Always include a comment at the top of configs explaining what changed
- Checkpoints save to `checkpoints/{run_name}/`
- Prompts for Mistral reflection live in `prompts/`
