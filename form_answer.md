We built 7 custom Claude Code skills that form a fully automated self-improvement loop for training a JAX-based LightGCN recommender system on the Gowalla dataset. The agent orchestrates the entire research cycle — from diagnosing training runs to applying fixes to generating final reports — using W&B MCP, Weave, and Mistral as its backbone tools.

**The core loop uses 3 skills:**

`/analyze-run` queries W&B MCP via GraphQL (`query_wandb_tool`) to fetch the latest training metrics (BPR loss, Recall@20, NDCG@20), diagnoses the model state (underfitting, overfitting, capacity limit, plateau), compares against LightGCN paper benchmarks, and recommends the next concrete action.

`/improve-model` takes that diagnosis and edits the Hydra YAML config in-place — tuning hyperparameters like learning rate, embedding dimension, number of GCN layers, regularization weight, dropout, batch size, and negative sampling count. It can also modify source code to add architectural improvements (layer attention weights, edge dropout, popularity-biased sampling). A more structured variant, `/apply-fix`, offers 5 pre-defined fixes (A–E) for common failure modes.

`/ablation` runs systematic Hydra multirun sweeps (`python src/main.py -m param=val1,val2,val3`), auto-tags each run in W&B, fetches results, and picks the winning configuration.

**Between iterations, `/weave-eval`** runs traced evaluations using W&B Weave with 4 custom scorers (recall, NDCG, diversity, coverage). It supports A/B testing two checkpoints by randomly splitting test users into groups — simulating a production traffic split to compare models head-to-head.

**When the loop converges, 2 reporting skills finalize everything:**

`/eval-report` runs a full Weave evaluation on the best checkpoint (500 users), calls the Mistral API for LLM-powered reflection on the results (Weave-traced), then uses `create_wandb_report_tool` via W&B MCP to generate a polished W&B Report covering the executive summary, self-improvement journey (iteration by iteration), test metrics, Mistral's analysis, architecture details, and key learnings.

`/project-changelog` generates a comprehensive audit trail as a W&B Report — pulling git commit history (`git log --stat -p`), config evolution across all experiment YAML versions, all W&B training runs with configs and metrics, and all Weave evaluation traces including Mistral reflections.

**Specific tools and integrations used:**
- **W&B MCP**: `query_wandb_tool` (GraphQL for runs, metrics, configs), `query_weave_traces_tool` (evaluation + reflection traces), `create_wandb_report_tool` (auto-generated markdown reports)
- **W&B Weave**: Traced evaluations with custom scorers, A/B test framework, Mistral reflection tracing
- **Mistral API**: LLM reflection on aggregate evaluation results (prompted via `prompts/reflection.txt`)
- **Hydra + OmegaConf**: YAML config management with CLI overrides and multirun sweeps for ablations
- **Git**: Full version control of config and code changes, used by `/project-changelog` for diff extraction

The workflow: Train → `/weave-eval` → `/analyze-run` → `/improve-model` | `/apply-fix` | `/ablation` → Train → repeat → `/eval-report` → `/project-changelog`. Every skill is a Claude Code slash command, and the agent drives the full cycle autonomously.
