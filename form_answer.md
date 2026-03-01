I built 6 custom Claude Code skills that form a fully automated self-improvement loop for training a JAX-based LightGCN recommender system on the Gowalla dataset. The agent orchestrates the entire research cycle from diagnosing training runs to applying fixes to generating final reports using W&B MCP, Weave, and Mistral as its backbone tools.

The core loop has 3 skills:

/analyze-run queries W&B MCP via GraphQL (query_wandb_tool) to fetch the latest training metrics (BPR loss, Recall@20, NDCG@20), diagnoses the model state (underfitting, overfitting, capacity limit, plateau), compares against LightGCN paper benchmarks, and recommends the next concrete action.

/apply-fix takes that diagnosis and implements exactly one focused change at a time. It offers 5 pre-defined fixes (A–E) for common failure modes: A) reg_weight tuning for overfitting, B) multiple negative samples for stronger gradients, C) learnable layer attention weights to address over-smoothing, D) edge dropout for regularization, E) popularity-biased negative sampling for harder negatives. It edits the Hydra YAML config in-place and can modify source code when needed.

/ablation runs systematic Hydra multirun sweeps (python src/main.py -m param=val1,val2,val3), auto-tags each run in W&B, fetches results, and picks the winning configuration. Useful when /analyze-run is inconclusive about which direction to go.

Between iterations, /weave-eval runs traced evaluations using W&B Weave with 4 custom scorers (recall, NDCG, diversity, coverage). It supports A/B testing two checkpoints by randomly splitting test users into groups — simulating a production traffic split to compare models head-to-head. This is especially useful for recommender systems in which we would like to see some behavior before full deployment in production.

Once the researcher is satisfied with the results obtained, 2 reporting skills keep track of the progress done across the project:

/eval-report is a 4-step pipeline that ties together the entire self-improvement journey into one artifact. First, it runs the Weave evaluation on the best checkpoint — generating top-20 recommendations for 500 test users scored by recall, NDCG, diversity, and coverage, with Mistral reflecting on the aggregate results (all Weave-traced). Second, it gathers all historical training data by querying W&B MCP with a GraphQL query that fetches every run's name, state, summaryMetrics, and config. Third, it pulls evaluation results and Mistral reflections from Weave using query_weave_traces_tool (filtering by op_name "Evaluation.evaluate" and "mistral_reflect"). Finally, it programmatically builds a W&B Report using wandb_workspaces — complete with PanelGrid containing LinePlots for Recall@20, NDCG@20, and BPR Loss across runs, filtered to the lgcn_gowalla_full run group. The report is structured as: executive summary, iteration-by-iteration self-improvement journey, Weave test-set evaluation, Mistral's LLM reflection, architecture details, and key learnings (what worked, what didn't, remaining gap to benchmarks).

/project-changelog generates a comprehensive audit trail as a W&B Report — pulling git commit history, config evolution across all experiment YAML versions, all W&B training runs with configs and metrics, and all Weave evaluation traces including Mistral reflections.

Specific tools and integrations:
1) W&B MCP: query_wandb_tool (GraphQL for runs, metrics, configs), query_weave_traces_tool (evaluation + reflection traces), create_wandb_report_tool (auto-generated markdown reports)
2) W&B Weave: Traced evaluations with custom scorers, A/B test framework, Mistral reflection tracing
3) Hydra + OmegaConf: YAML config management with CLI overrides and multirun sweeps for ablations

The workflow: Train → /analyze-run → /apply-fix | /ablation → Train → ... → /eval-report → /project-changelog → /weave-eval. Each skill is a Claude Code slash command. The agent invokes W&B MCP for metrics, Weave for traced evals, and Mistral for LLM reflection — fully automated.
