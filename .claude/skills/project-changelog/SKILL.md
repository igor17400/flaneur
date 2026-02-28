---
name: project-changelog
description: Generate a W&B Report tracking all Claude Code modifications — git commits, config evolution, code changes, training metrics, and Weave evaluations — as a full project changelog.
allowed-tools: Bash, Read, Grep, Glob
---

# Project Changelog

Generate a comprehensive W&B Report documenting every modification Claude Code has made to the project. This gives a time-ordered view of the entire self-improvement journey.

## Step 1: Gather Git History

Collect all commits to understand what changed and when:

```bash
# Full commit log with stats
git log --oneline --stat --no-merges

# Detailed diffs for source and config changes
git log --no-merges -p -- src/ configs/
```

Parse each commit into a structured entry:
- **Hash + date**: when it happened
- **Message**: what was intended
- **Files changed**: which files were touched
- **Diff summary**: key lines added/removed (focus on src/ and configs/)

## Step 2: Track Config Evolution

Read all experiment configs and diff them to show the hyperparameter trajectory:

```bash
ls configs/experiment/lgcn_gowalla*.yaml
```

For each config file, extract:
- `embed_dim`, `n_layers`, `lr`, `reg_weight`, `embed_dropout`, `batch_size`, `epochs`, `patience`
- The comment at the top explaining what changed

Build a comparison table showing how each parameter evolved across versions.

## Step 3: Gather W&B Training Metrics

Use the wandb Python API to fetch all runs:

```python
import wandb
api = wandb.Api()
runs = api.runs("igorlima1740/flaneur", order="+created_at")
for r in runs:
    print(r.name, r.state, r.created_at, r.summary, dict(r.config))
```

Extract per-run:
- Run name, state, creation time
- Best val Recall@20, best val NDCG@20
- Final BPR loss
- Full config

## Step 4: Gather Weave Evaluation Results

Query Weave for all evaluation traces:

```python
import weave
client = weave.init("igorlima1740/flaneur")
calls = client.get_calls(
    filter={"op_names": ["weave:///igorlima1740/flaneur/op/Evaluation.evaluate:*"]},
    limit=20
)
```

And Mistral reflections:

```python
calls = client.get_calls(
    filter={"op_names": ["weave:///igorlima1740/flaneur/op/mistral_reflect:*"]},
    limit=10
)
```

Extract per-evaluation:
- Timestamp
- Test Recall@20, NDCG@20, diversity, coverage
- Mistral reflection summary (diagnosis + recommendations)

## Step 5: Detect Code-Level Changes

Identify structural changes to the model, training loop, and data pipeline:

```bash
# Show all unique files modified across all commits
git log --name-only --pretty=format: -- src/ | sort -u

# For each source file, get the full diff history
git log -p -- src/model.py
git log -p -- src/train.py
git log -p -- src/data.py
```

Summarize code changes as:
- **What function/class changed**
- **Before vs after** (brief)
- **Why** (from commit message or config comment)

## Step 6: Generate W&B Report

Use `wandb_workspaces` to create the report. Entity: `igorlima1740`, Project: `flaneur`.

Structure the report with these sections:

```markdown
# Flaneur: Project Changelog

[TOC]

## Overview
- Project: LightGCN on Gowalla (JAX)
- Total commits: {N}
- Training runs: {N}
- Weave evaluations: {N}
- Date range: {first_commit} to {latest_commit}

## Timeline

### {Date}: {Commit message or iteration name}
- **Commit**: {hash} — {message}
- **Files changed**: {list}
- **Key changes**: {summary of code/config diffs}
- **Config**: {hyperparameter values if config was changed}
- **Training result**: Recall@20={}, NDCG@20={} (if a run followed)
- **Status**: {improved / regressed / baseline}

(Repeat for each significant change, in chronological order)

## Config Evolution Table
| Version | embed_dim | n_layers | lr | reg_weight | dropout | epochs | patience | Val Recall@20 | Val NDCG@20 |
|---------|-----------|----------|----|------------|---------|--------|----------|---------------|-------------|
| baseline | 64 | 3 | 1e-3 | 1e-4 | 0.0 | 50 | - | 0.1438 | 0.0749 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Code Changes Log
### model.py
- {date}: {description of change and why}

### train.py
- {date}: {description of change and why}

### data.py
- {date}: {description of change and why}

## Metrics Trajectory
(Include PanelGrid with LinePlots for val/recall@20, val/ndcg@20, train/bpr_loss across all runs)

## Weave Evaluations
| Date | Checkpoint | Test Recall@20 | Test NDCG@20 | Diversity | Mistral Diagnosis |
|------|-----------|----------------|--------------|-----------|-------------------|
| ... | ... | ... | ... | ... | ... |

## Cumulative Impact
- Starting point: Recall@20={baseline}, NDCG@20={baseline}
- Current best: Recall@20={best}, NDCG@20={best}
- Total improvement: +{X}% Recall, +{Y}% NDCG
- Distance to benchmark: {analysis}

## Tools & Infrastructure
- Claude Code skills used: /analyze-run, /improve-model, /eval-report, /project-changelog
- W&B for experiment tracking and reporting
- Weave for evaluation tracing and LLM reflection
```

Use `wandb_workspaces.reports.v2` to build the report programmatically:
- Use `wr.H1`, `wr.H2`, `wr.H3` for headings
- Use `wr.P` for paragraphs
- Use `wr.UnorderedList` / `wr.OrderedList` with plain strings
- Use `wr.PanelGrid` with `wr.LinePlot` for training curves
- Use `wr.MarkdownBlock` for tables (rendered as markdown)
- Lists take plain strings, NOT ListItem objects

After saving, print the report URL.
