---
name: analyze-run
description: Query W&B MCP to analyze the latest LightGCN training runs. Use this after a training run completes to understand performance, diagnose issues (underfitting, overfitting, convergence), and identify what to change next.
allowed-tools: Bash, Read, Grep, Glob
---

# Analyze Training Run

You are analyzing LightGCN training runs on the Gowalla dataset using W&B MCP tools.

## Reference Benchmarks (LightGCN paper, Gowalla)
- Recall@20: ~0.183
- NDCG@20: ~0.154

## Step 1: Fetch Latest Runs

Use `query_wandb_tool` to get recent runs from entity `igorlima1740`, project `flaneur`:

```graphql
query GetRuns($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    runs(first: 10, order: "-createdAt") {
      edges {
        node { name displayName state createdAt summaryMetrics }
      }
      pageInfo { endCursor hasNextPage }
    }
  }
}
```

## Step 2: Fetch Training History

For each run of interest, get the loss curve and eval metrics using `sampledHistory`:

```graphql
query RunHistory($entity: String!, $project: String!, $runId: String!, $specs: [JSONString!]!) {
  project(name: $project, entityName: $entity) {
    run(name: $runId) {
      historyKeys
      sampledHistory(specs: $specs)
    }
  }
}
```

Use specs: `["{\"keys\": [\"train/bpr_loss\", \"val/recall@20\", \"val/ndcg@20\"], \"samples\": 100}"]`

## Step 3: Diagnose

Analyze the metrics and classify the run:

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Loss still decreasing, metrics still improving | Underfitting / needs more epochs | Increase epochs or reduce lr for finer convergence |
| Loss plateaued, metrics plateaued | Capacity limit reached | Increase embed_dim, add layers, or change architecture |
| Loss decreasing but metrics flat/declining | Overfitting | Increase reg_weight, add dropout, reduce embed_dim |
| Metrics far below benchmarks after convergence | Suboptimal hyperparameters | Tune lr, reg_weight, n_layers, embed_dim |
| Metrics near benchmarks | Good — diminishing returns | Try fine-grained tuning or architectural changes |

## Step 4: Compare Runs

If multiple runs exist, compare them:
- Which config achieved better Recall@20?
- What changed between configs?
- Is the improvement statistically meaningful?

## Step 5: Output

Produce a structured analysis:
1. **Run summary**: name, epochs, final metrics, config
2. **Diagnosis**: underfitting/overfitting/converged
3. **Loss curve shape**: monotone decreasing? plateau? oscillating?
4. **Comparison** to benchmarks and previous runs
5. **Recommended next action**: specific config/code changes with reasoning
