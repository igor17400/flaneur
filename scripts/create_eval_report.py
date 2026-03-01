"""Generate comprehensive W&B Report: ablation studies, A/B test, and Weave evaluation."""

import base64

import wandb
from wandb_workspaces.reports.v1 import Report
from wandb_workspaces.reports.v1._blocks import (
    H1,
    H2,
    H3,
    HorizontalRule,
    Image,
    MarkdownBlock,
    P,
    PanelGrid,
)
from wandb_workspaces.reports.v1.runset import Runset

ENTITY = "igorlima1740"
PROJECT = "flaneur"


def mermaid_to_image_url(mermaid_def: str) -> str:
    encoded = base64.urlsafe_b64encode(mermaid_def.encode("utf-8")).decode("ascii")
    return f"https://mermaid.ink/img/{encoded}?type=png&bgColor=white"


WORKFLOW_MERMAID = r"""
flowchart TD
    START([Start]) --> ANALYZE1["/analyze-run"]
    ANALYZE1 --> DIAGNOSE{Diagnosis}
    DIAGNOSE -->|Suboptimal HP| ABLATION["/ablation\n(Hydra multirun)"]
    DIAGNOSE -->|Code change needed| IMPROVE["/improve-model"]
    ABLATION --> TRAIN["Train"]
    IMPROVE --> TRAIN
    TRAIN --> WEAVE["/weave-eval"]
    WEAVE --> ANALYZE2["/analyze-run"]
    ANALYZE2 --> IMPROVED{Improved?}
    IMPROVED -->|No| DIAGNOSE
    IMPROVED -->|Yes| NEXT{More to tune?}
    NEXT -->|Yes| ABLATION
    NEXT -->|No| FINAL["Final long run"]
    FINAL --> EVAL["/eval-report"]
    EVAL --> DONE([Done])
    style ABLATION fill:#e74c3c,stroke:#c0392b,color:#fff
    style WEAVE fill:#1abc9c,stroke:#16a085,color:#fff
    style TRAIN fill:#3498db,stroke:#2980b9,color:#fff
    style FINAL fill:#9b59b6,stroke:#8e44ad,color:#fff
"""


def create_report():
    api = wandb.Api()
    flowchart_url = mermaid_to_image_url(WORKFLOW_MERMAID)

    report = Report(
        project=PROJECT,
        entity=ENTITY,
        title="Flaneur: LightGCN Self-Improvement Report",
        description="Ablation studies, A/B testing, and Weave evaluation for LightGCN on Gowalla",
        width="readable",
        blocks=[
            # ── Executive Summary ──
            H1(text="Flaneur: LightGCN Self-Improvement Report"),
            MarkdownBlock(text=[
                "**Task:** Collaborative filtering on the Gowalla check-in dataset (29,858 users, 40,981 items)\n\n",
                "**Model:** LightGCN (JAX implementation) with BPR loss and cosine LR schedule\n\n",
                "**Self-improvement:** Automated loop driven by Claude Code skills — `/analyze-run`, `/ablation`, `/weave-eval`, `/eval-report`\n\n",
                "**Best result (100 epochs):** Recall@20 = 0.1645, NDCG@20 = 0.0857 (embed_dim=256)\n\n",
                "**Weave test evaluation:** Test Recall@20 = 0.1355, Test NDCG@20 = 0.1060, Diversity = 0.167\n\n",
                "**Paper benchmark:** Recall@20 ≈ 0.183, NDCG@20 ≈ 0.154",
            ]),

            HorizontalRule(),

            # ── Workflow ──
            H2(text="Self-Improvement Workflow"),
            P(text=[
                "The project uses Claude Code skills in a loop: analyze metrics → diagnose → ",
                "ablate or improve → train → evaluate with Weave → repeat.",
            ]),
            Image(
                url=flowchart_url,
                caption="Automated self-improvement loop with Claude Code skills",
            ),

            HorizontalRule(),

            # ── Ablation 1: embed_dim ──
            H2(text="Ablation Study 1: Embedding Dimension"),
            MarkdownBlock(text=[
                "**Hypothesis:** Larger embeddings capture more complex user-item interactions.\n\n",
                "**Setup:** Sweep `embed_dim` over [64, 128, 256], all other params fixed (4 layers, lr=1e-3, reg=1e-5, n_negatives=3).\n\n",
                "#### Results at 100 epochs\n\n",
                "| embed_dim | Val Recall@20 | Val NDCG@20 | BPR Loss | Δ Recall |\n",
                "|-----------|--------------|-------------|----------|----------|\n",
                "| 64        | 0.1417       | 0.0738      | 0.0413   | baseline |\n",
                "| 128       | 0.1535       | 0.0798      | 0.0322   | +8.3%    |\n",
                "| **256**   | **0.1645**   | **0.0857**  | 0.0244   | **+16.1%** |\n\n",
                "#### Results at 30 epochs (fast ablation)\n\n",
                "| embed_dim | Val Recall@20 | Val NDCG@20 | Δ Recall |\n",
                "|-----------|--------------|-------------|----------|\n",
                "| 64        | 0.1133       | 0.0595      | baseline |\n",
                "| 128       | 0.1208       | 0.0636      | +6.6%    |\n",
                "| **256**   | **0.1287**   | **0.0677**  | **+13.6%** |\n\n",
                "**Conclusion:** Consistent scaling benefit from larger embeddings. dim=256 wins at both 30 and 100 epochs. ",
                "Relative ranking is preserved at 30 epochs, validating the fast ablation strategy.\n\n",
                "**Winner: embed_dim = 256**",
            ]),

            HorizontalRule(),

            # ── Ablation 2: reg_weight ──
            H2(text="Ablation Study 2: L2 Regularization"),
            MarkdownBlock(text=[
                "**Hypothesis:** With dim=256, stronger regularization may reduce overfitting to popular items.\n\n",
                "**Setup:** Sweep `reg_weight` over [1e-5, 1e-4, 1e-3] with embed_dim=256 fixed.\n\n",
                "#### Results at 30 epochs\n\n",
                "| reg_weight | Val Recall@20 | Val NDCG@20 | BPR Loss | Δ Recall |\n",
                "|------------|--------------|-------------|----------|----------|\n",
                "| **1e-5**   | **0.1287**   | **0.0677**  | 0.0558   | **best** |\n",
                "| 1e-4       | 0.1284       | 0.0676      | 0.0605   | -0.2%    |\n",
                "| 1e-3       | 0.1241       | 0.0651      | 0.0937   | -3.6%    |\n\n",
                "**Conclusion:** reg_weight has minimal impact between 1e-5 and 1e-4 — both produce nearly identical results. ",
                "1e-3 is clearly too strong, causing underfitting (higher loss, lower metrics). ",
                "L2 regularization is not a bottleneck at this configuration.\n\n",
                "**Winner: reg_weight = 1e-5 (current default)**",
            ]),

            HorizontalRule(),

            # ── Weave Evaluation ──
            H2(text="Weave Evaluation: Test Set Performance"),
            MarkdownBlock(text=[
                "Weave evaluations run the trained model on held-out test users, scoring with recall, NDCG, diversity, and coverage.\n\n",
                "#### Best Model (dim=256, reg=1e-5, 100 epochs, 100 test users)\n\n",
                "| Metric | Value | Paper Benchmark | Gap |\n",
                "|--------|-------|-----------------|-----|\n",
                "| Test Recall@20 | 0.1613 | ~0.183 | -12% |\n",
                "| Test NDCG@20 | 0.1330 | ~0.154 | -14% |\n",
                "| Diversity | 0.249 | — | Low |\n",
                "| Avg Item Popularity | 206.8 | — | High (popularity bias) |\n",
                "| Cold Items Recommended | 0.0 | — | None |\n",
                "| Latency | 28ms | — | Good |\n\n",
                "**Key insight:** The model finds relevant items (decent recall) but exhibits strong popularity bias — ",
                "zero cold items recommended and low diversity (0.249). The NDCG gap to benchmark (-14%) suggests ranking quality needs improvement.",
            ]),

            HorizontalRule(),

            # ── A/B Test ──
            H2(text="A/B Test Simulation"),
            MarkdownBlock(text=[
                "Simulated a production A/B test with **non-overlapping user groups** — each user sees only one model.\n\n",
                "#### Setup\n",
                "- **Group A** (200 users, IDs 158–14631) → dim=256 model\n",
                "- **Group B** (200 users, IDs 14676–29772) → dim=128 model\n",
                "- No user overlap, simulating real traffic splitting\n\n",
                "#### Results\n\n",
                "| Metric | Group A (dim=256) | Group B (dim=128) | Diff |\n",
                "|--------|-------------------|-------------------|------|\n",
                "| Recall@20 | 0.1259 | **0.1440** | B +14.4% |\n",
                "| NDCG@20 | 0.1150 | **0.1186** | B +3.1% |\n",
                "| Diversity | **0.196** | 0.157 | A +24.8% |\n",
                "| Hits/user | **0.94** | 0.79 | A +19.0% |\n",
                "| Test items/user | 9.52 | 6.59 | Different |\n\n",
                "#### Simpson's Paradox\n\n",
                "The results appear to **contradict** the controlled comparison (where dim=256 wins on recall). ",
                "This is because the user populations are not equivalent:\n\n",
                "- Group A users have **more test items** (9.52 vs 6.59), making recall harder to achieve\n",
                "- Group A actually gets **more raw hits** (0.94 vs 0.79)\n",
                "- The recall metric penalizes users with larger ground truth sets\n\n",
                "**Takeaway:** In real A/B testing, stratified sampling or propensity score matching is essential. ",
                "Raw metric comparison on non-equivalent populations can be misleading. ",
                "The controlled `--compare` evaluation (same users, both models) is more reliable for model selection.",
            ]),

            HorizontalRule(),

            # ── Mistral Reflection ──
            H2(text="Mistral Reflection"),
            MarkdownBlock(text=[
                "After each Weave evaluation, Mistral analyzes the results and suggests improvements. Key recommendations:\n\n",
                "1. **Increase embed_dim to 512** — scaling curve hasn't saturated\n",
                "2. **Add embed_dropout = 0.2** — regularize to improve generalization\n",
                "3. **Increase n_negatives to 5** — more contrastive signal for ranking quality (NDCG)\n",
                "4. **Lower learning rate to 5e-4** — finer convergence at higher capacity\n",
                "5. **Increase reg_weight to 1e-4** — ablation showed minimal impact, but Mistral flags the popularity bias\n\n",
                "These recommendations align with the gap analysis: the model needs better **ranking quality** (NDCG) ",
                "and **diversity**, not just raw recall.",
            ]),

            HorizontalRule(),

            # ── Key Learnings ──
            H2(text="Key Learnings"),
            H3(text="What Worked"),
            MarkdownBlock(text=[
                "- **Embedding dimension scaling** — consistent +8% per doubling (64→128→256)\n",
                "- **Fast ablation strategy** — 30-epoch runs preserve relative ranking, 3x faster than 100-epoch\n",
                "- **Weave Model class** — tracking config per evaluation enables meaningful leaderboard comparison\n",
                "- **Per-config checkpoints** — descriptive naming (`dim256_layers4_...`) makes comparison straightforward\n",
                "- **Descriptive W&B run names + grouping** — instantly scannable experiment history\n",
            ]),
            H3(text="What Didn't Help"),
            MarkdownBlock(text=[
                "- **Stronger L2 regularization (1e-3)** — caused underfitting, -3.6% recall\n",
                "- **A/B test with random split** — Simpson's paradox made results misleading without stratification\n",
            ]),
            H3(text="Remaining Gap to Benchmarks"),
            MarkdownBlock(text=[
                "| Metric | Current Best | Paper Benchmark | Gap |\n",
                "|--------|-------------|-----------------|-----|\n",
                "| Recall@20 | 0.1645 (val) / 0.1613 (test) | ~0.183 | ~10-12% |\n",
                "| NDCG@20 | 0.0857 (val) / 0.1330 (test) | ~0.154 | ~14% |\n\n",
                "**Next steps to close the gap:**\n",
                "1. Try embed_dim=512\n",
                "2. Add embedding dropout (0.1–0.2)\n",
                "3. Increase n_negatives to 5\n",
                "4. Run final 1000-epoch training with best settings\n",
                "5. Consider architectural changes (layer norm, DirectAU loss)\n",
            ]),

            HorizontalRule(),

            # ── Tools Used ──
            H2(text="Tools Used"),
            MarkdownBlock(text=[
                "| Tool | Purpose |\n",
                "|------|--------|\n",
                "| **W&B Experiment Tracking** | Loss curves, metric logging, run comparison, grouping |\n",
                "| **W&B Weave** | Test-set evaluation, per-user traces, model versioning, leaderboard |\n",
                "| **W&B Reports** | This report — documenting the full journey |\n",
                "| **Weave Leaderboard** | Side-by-side model comparison with full config visibility |\n",
                "| **Hydra Multirun** | Parameter sweeps for ablation studies |\n",
                "| **Mistral Reflection** | AI-driven analysis of evaluation results |\n",
                "| **Claude Code Skills** | `/analyze-run`, `/ablation`, `/weave-eval`, `/eval-report` |\n\n",
                "**Weave links:**\n",
                "- [Leaderboard](https://wandb.ai/igorlima1740/flaneur/weave/leaderboards/LightGCN-Gowalla)\n",
                "- [Traces](https://wandb.ai/igorlima1740/flaneur/weave)\n",
            ]),
        ],
    )

    report.save()
    print(f"Report created: {report.url}")
    return report.url


if __name__ == "__main__":
    create_report()
