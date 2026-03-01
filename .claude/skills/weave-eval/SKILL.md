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

## Default: A/B test the two best trained models

The default invocation is an **A/B test comparing the two best trained checkpoints** on randomly assigned, non-overlapping user groups — simulating a real production traffic split where each user only sees one model.

### Step 1: Identify the two best checkpoints

List checkpoints sorted by val recall:

```bash
uv run python src/evaluate_weave.py --list
```

Pick the **top 2 by Val Recall@20** that differ meaningfully (e.g. different embed_dim, different reg_weight). If the top two are nearly identical configs, pick the best and the next-best with a **different architecture** (e.g. d256 vs d128) for a more informative comparison.

### Step 2: Ensure prediction files exist for both

```bash
ls predictions/
```

If either prediction file is missing, generate it:

```bash
uv run python src/infer.py --run <checkpoint_name>
```

### Step 3: Run A/B test

Randomly assign 200 users to each group, each group sees only one model:

```bash
uv run python src/evaluate_weave.py --ab-test <model_a> <model_b> --from-predictions --n_users 200
```

This samples 400 users total (seed=42), splits into two non-overlapping groups of 200, and evaluates each model on its own group — just like real traffic splitting.

### Step 4: Read the results

The evaluation prints a JSON summary for each group with these metrics:

| Metric | What it tells you |
|--------|-------------------|
| `recall_scorer.recall.mean` | Fraction of test items found in top-20 |
| `ndcg_scorer.ndcg.mean` | Ranking quality of recommendations |
| `diversity_scorer.diversity.mean` | How varied the recommendations are (1 = max diverse) |
| `coverage_scorer.avg_item_popularity.mean` | Whether model recommends popular or niche items |
| `coverage_scorer.cold_items_recommended.mean` | How many cold-start items appear in recs |

### Step 5: Report findings

Present a comparison table and recommend which model to deploy:

```
| Group | Model                     | Recall@20 | NDCG@20 | Diversity | Avg Popularity |
|-------|---------------------------|-----------|---------|-----------|----------------|
| A     | LightGCN-d256-L4-reg1e-05 | ...       | ...     | ...       | ...            |
| B     | LightGCN-d128-L4-reg1e-05 | ...       | ...     | ...       | ...            |
```

## Alternative modes

```bash
# Compare on SAME users (controlled, not A/B)
uv run python src/evaluate_weave.py --from-predictions <name1> <name2> --n_users 200

# Single model evaluation
uv run python src/evaluate_weave.py --from-predictions <name> --n_users 200

# From checkpoint directly (recomputes predictions)
uv run python src/evaluate_weave.py --run <checkpoint_name> --n_users 100
```

## Trace naming

Traces in Weave UI use human-readable names including the distinguishing config:
- `LightGCN-d256-L4-reg1e-05 (200 users)`
- `A/B Group A: LightGCN-d256-L4-reg1e-05 (200 users)`
- `A/B Group B: LightGCN-d128-L4-reg1e-05 (200 users)`

## Compare in Weave UI

All evaluations are logged to the Weave project `flaneur`:

1. **Leaderboard tab** — side-by-side comparison of all evaluated models
2. **Traces tab** — drill into individual user recommendations
3. **Filter by eval name** — each eval has a descriptive name

## Workflow Position

```
Train → /weave-eval → /analyze-run → /improve-model or /ablation → Train → ...
```
