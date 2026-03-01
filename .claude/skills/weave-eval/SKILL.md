---
name: weave-eval
description: Run a quick Weave evaluation on the latest checkpoint to get test-set signals (recall, NDCG, diversity, coverage) without Mistral reflection. Use after any training run or ablation to compare candidates on richer metrics before deciding next steps.
allowed-tools: Bash, Read, Grep, Glob
---

# Quick Weave Evaluation

Lightweight Weave eval that runs **after any training run** — not just at the end of the loop. Gives you test-set recall, NDCG, diversity, and coverage to feed into `/analyze-run`.

## When to Use

- After a single training run to check quality beyond val metrics
- After an ablation sweep to compare candidates on the full scorer suite
- Before deciding what to ablate or improve next
- Any time you want test-set signals without the overhead of `/eval-report`

## Step 1: Identify the Checkpoint

List available checkpoints:

```bash
uv run python src/evaluate_weave.py --list
```

Pick the most recent one, or the one matching the run you want to evaluate.

## Step 2: Run Fast Evaluation

Run Weave eval with a small user sample (100-200) and **no Mistral reflection** for speed:

```bash
# Fast eval: 100 users, ~2 min
uv run python src/evaluate_weave.py --run lgcn_gowalla_full --n_users 100
```

**Note:** The script always runs Mistral reflection at the end. To skip it for speed, you can set `MISTRAL_API_KEY` to empty — the reflection will fail gracefully and the core metrics still get logged to Weave.

For ablation comparison, run eval on each checkpoint sequentially. Each run is traced separately in Weave and updates the leaderboard.

## Step 3: Read the Results

The evaluation prints a JSON summary with these metrics:

| Metric | What it tells you |
|--------|-------------------|
| `recall_scorer.recall.mean` | Fraction of test items found in top-20 |
| `ndcg_scorer.ndcg.mean` | Ranking quality of recommendations |
| `diversity_scorer.diversity.mean` | How varied the recommendations are (1 = max diverse) |
| `coverage_scorer.avg_item_popularity.mean` | Whether model recommends popular or niche items |
| `coverage_scorer.cold_items_recommended.mean` | How many cold-start items appear in recs |

## Step 4: Compare in Weave UI

All evaluations are logged to the Weave project `flaneur`. You can:

1. **Leaderboard tab** — side-by-side comparison of all evaluated checkpoints
2. **Traces tab** — drill into individual user recommendations
3. **Filter by eval name** — each eval is named `eval_{checkpoint_name}`

## Step 5: Report Findings

Present a comparison table to guide the next decision:

```
| Config          | Val Recall@20 | Test Recall@20 | Diversity | Coverage |
|-----------------|---------------|----------------|-----------|----------|
| embed_dim=64    | 0.1417        | ...            | ...       | ...      |
| embed_dim=128   | 0.1528        | ...            | ...       | ...      |
| embed_dim=256   | ...           | ...            | ...       | ...      |
```

Key insight: Val metrics only measure recall/NDCG. Weave eval adds **diversity and coverage**, which may reveal that the "best" config by recall actually recommends a narrow set of popular items.

## Workflow Position

```
Train → /weave-eval → /analyze-run → /improve-model or /ablation → Train → ...
```

This skill sits between training and analysis, enriching the signal available to `/analyze-run` with test-set quality metrics.
