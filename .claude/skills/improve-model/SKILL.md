---
name: improve-model
description: Apply improvements to the LightGCN model based on analysis from /analyze-run. Edits the experiment config in-place, modifies code if needed, and prepares the next training run.
allowed-tools: Read, Edit, Write, Bash, Glob, Grep
---

# Improve Model

Apply concrete improvements to LightGCN based on run analysis. This skill edits the existing experiment config in-place and optionally modifies source code.

## Project Structure

- Configs: `configs/experiment/*.yaml` (Hydra)
- Model: `src/model.py` (LightGCN forward pass, BPR loss)
- Training: `src/train.py` (training loop, optimizer, eval)
- Data: `src/data.py` (dataset loading, adjacency matrix, negative sampling)

## Step 1: Read Current Config

Read the experiment config `configs/experiment/lgcn_gowalla_full.yaml` and source code to understand the current state.

## Step 2: Determine Changes

Based on the analysis, select from these improvement levers:

### Hyperparameter Tuning (config only)
| Param | Config Key | Range | Notes |
|-------|-----------|-------|-------|
| Embedding dim | `model.embed_dim` | 32, 64, 128, 256 | Higher = more capacity, more memory |
| GCN layers | `model.n_layers` | 1-4 | >4 risks over-smoothing |
| Learning rate | `train.lr` | 1e-4 to 1e-2 | Combined with cosine schedule |
| L2 regularization | `train.reg_weight` | 1e-6 to 1e-2 | Controls overfitting |
| Batch size | `train.batch_size` | 1024, 2048, 4096 | Affects gradient noise |
| Embedding dropout | `model.embed_dropout` | 0.0-0.3 | Regularization |
| Early stopping | `train.patience` | 50-200, 0=off | Prevents overtraining |

### Code-Level Improvements (modify source)
- **Negative sampling**: popularity-biased sampling in `data.py`
- **Layer attention**: learnable layer weights instead of uniform mean in `model.py`
- **Edge dropout**: randomly drop adjacency edges during training in `model.py`
- **Xavier init**: scale init by 1/sqrt(embed_dim) in `model.py`
- **Multiple negatives**: sample K negatives per positive in `data.py`

## Step 3: Edit Experiment Config In-Place

Edit `configs/experiment/lgcn_gowalla_full.yaml` directly. Update the comment at the top to explain what changed and why. Keep `run_name: lgcn_gowalla_full` — W&B tracks each run separately even with the same name.

**Important**: Do NOT create new config files (v2, v3, etc.). Always edit `lgcn_gowalla_full.yaml` in-place. W&B and git history track the evolution.

## Step 4: Apply Code Changes (if needed)

If the improvement requires code changes, edit the relevant source files. Always:
- Keep changes minimal and focused
- Don't break backward compatibility with existing configs
- Use `getattr(cfg.model, "new_param", default)` for new config params

## Step 5: Output

Provide:
1. **What changed**: list of param/code changes with reasoning
2. **Config file**: path to the edited experiment config
3. **Run command**: exact command to start training
4. **Expected outcome**: what metric improvement to expect and why

Run command:
```bash
uv run python src/main.py experiment=lgcn_gowalla_full
```
