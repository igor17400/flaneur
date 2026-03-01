---
name: eval-report
description: Run Weave evaluation on a checkpoint and generate a W&B Report summarizing the full self-improvement journey. Use this after the improvement loop is complete.
allowed-tools: Bash, Read, Grep, Glob
---

# Evaluate & Generate Report

Final step in the self-improvement loop. Run the Weave test evaluation on the best checkpoint, then generate a comprehensive W&B Report.

## Step 1: Run Weave Evaluation

Execute the Weave evaluation pipeline on the best checkpoint:

```bash
cd flaneur
uv run python src/evaluate_weave.py --run {best_checkpoint_name} --n_users 500
```

This will:
- Generate top-20 recommendations for 500 test users
- Score with recall, NDCG, diversity, and coverage scorers
- Have Mistral reflect on the aggregate results
- All traced in Weave

## Step 2: Gather All Run Data via W&B MCP

Use `query_wandb_tool` to fetch all training runs and their configs/metrics:

```graphql
query AllRuns($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    runs(first: 20, order: "-createdAt") {
      edges {
        node { name displayName state createdAt summaryMetrics config }
      }
      pageInfo { endCursor hasNextPage }
    }
  }
}
```

Entity: `igorlima1740`, Project: `flaneur`

## Step 3: Check Weave Evaluation Results

Use `query_weave_traces_tool` to fetch the latest evaluation results:

```python
query_weave_traces_tool(
    entity_name="igorlima1740",
    project_name="flaneur",
    filters={"op_name_contains": "Evaluation.evaluate"},
    columns=["id", "output", "started_at"],
    return_full_data=True,
    limit=5
)
```

Also fetch Mistral reflection:
```python
query_weave_traces_tool(
    entity_name="igorlima1740",
    project_name="flaneur",
    filters={"op_name_contains": "mistral_reflect"},
    columns=["id", "output", "started_at"],
    return_full_data=True,
    limit=5
)
```

## Step 4: Generate W&B Report

Use `wandb_workspaces` to build the report programmatically. Entity: `igorlima1740`, Project: `flaneur`.

**Important:** When creating `PanelGrid` with `Runset`, filter to only show runs in the `lgcn_gowalla_full` group:

```python
wr.PanelGrid(
    runsets=[wr.Runset(
        project="flaneur",
        entity="igorlima1740",
        filters='Group == "lgcn_gowalla_full"',
    )],
    panels=[
        wr.LinePlot(x="Step", y=["val/recall@20"], title="Val Recall@20"),
        wr.LinePlot(x="Step", y=["val/ndcg@20"], title="Val NDCG@20"),
        wr.LinePlot(x="Step", y=["train/bpr_loss"], title="BPR Loss"),
    ],
)
```

This ensures only training runs inside the `lgcn_gowalla_full` group appear in the charts, not unrelated runs.

Structure the report as follows:

```markdown
# Flaneur: Self-Improving LightGCN on Gowalla

[TOC]

## Executive Summary
- Task: Collaborative filtering on Gowalla check-in dataset
- Model: LightGCN (JAX implementation)
- Self-improvement: {N} iterations driven by Claude Code via W&B MCP
- Best result: Recall@20={value}, NDCG@20={value}
- Improvement over baseline: +{X}% Recall, +{Y}% NDCG

## Self-Improvement Journey

### Iteration 1: Baseline
- Config: embed_dim=64, lr=1e-3, n_layers=3
- Results: Recall@20={}, NDCG@20={}
- Diagnosis: {what the agent found}

### Iteration 2: {name}
- Changes: {what changed and why}
- Results: Recall@20={}, NDCG@20={}
- Improvement: +{X}% Recall

### Iteration N: {name}
...

## Weave Evaluation (Held-Out Test Set)
- Evaluated on {N} test users
- Test Recall@20: {}
- Test NDCG@20: {}
- Diversity: {}
- Coverage: {}

## Mistral Reflection
{Summary of Mistral's analysis from Weave traces}

## Architecture & Methodology
- LightGCN with {layers} layers, {dim}-dim embeddings
- BPR loss with L2 regularization
- Cosine LR schedule, embedding dropout, early stopping
- Train/Val/Test split (90/10 from train, test held out)

## Key Learnings
- What worked: {list improvements that helped}
- What didn't: {list changes that didn't improve metrics}
- Remaining gap to benchmarks: {analysis}

## Tools Used
- **W&B Models**: Experiment tracking, loss curves, metric logging
- **W&B Weave**: Online evaluation simulation, Mistral reflection traces
- **W&B MCP**: Agent-driven analysis, run comparison, report generation
- **Claude Code Skills**: /analyze-run, /improve-model, /eval-report
```
