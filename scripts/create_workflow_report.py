"""Create a W&B Report with the Flaneur self-improvement workflow flowchart."""

import base64
import urllib.parse

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
)

ENTITY = "igorlima1740"
PROJECT = "flaneur"

# Mermaid flowchart definition
MERMAID_DEF = r"""
flowchart TD
    START([Start]) --> ANALYZE1["/analyze-run\n(Query W&B metrics)"]

    ANALYZE1 --> DIAGNOSE{Diagnosis}
    DIAGNOSE -->|Underfitting| IMPROVE_CODE["/improve-model\n(Architecture or\nloss change)"]
    DIAGNOSE -->|Overfitting| IMPROVE_REG["/improve-model\n(Regularization,\ndropout, capacity)"]
    DIAGNOSE -->|Suboptimal HP| ABLATION["/ablation\n(Hydra multirun\n100ep sweep)"]

    IMPROVE_CODE --> TRAIN["Train\n(uv run python src/main.py)"]
    IMPROVE_REG --> TRAIN
    ABLATION --> COMPARE["Compare ablation\nresults in W&B"]
    COMPARE --> APPLY_WINNER["Apply winner\nto config"]
    APPLY_WINNER --> TRAIN

    TRAIN --> WEAVE["/weave-eval\n(Test-set: recall, NDCG,\ndiversity, coverage)"]
    WEAVE --> ANALYZE2["/analyze-run\n(Check improvement +\nWeave signals)"]

    ANALYZE2 --> IMPROVED{Improved?}
    IMPROVED -->|No| DIAGNOSE
    IMPROVED -->|Yes, more to gain| NEXT_ABLATION{More params\nto ablate?}
    NEXT_ABLATION -->|Yes| ABLATION
    NEXT_ABLATION -->|No| FINAL

    IMPROVED -->|Converged| FINAL["Final 1000-epoch run\nwith best settings"]
    FINAL --> EVAL["/eval-report\n(Full Weave eval +\nW&B Report)"]
    EVAL --> DONE([Done])

    style START fill:#2ecc71,stroke:#27ae60,color:#fff
    style DONE fill:#2ecc71,stroke:#27ae60,color:#fff
    style DIAGNOSE fill:#f39c12,stroke:#e67e22,color:#fff
    style IMPROVED fill:#f39c12,stroke:#e67e22,color:#fff
    style NEXT_ABLATION fill:#f39c12,stroke:#e67e22,color:#fff
    style TRAIN fill:#3498db,stroke:#2980b9,color:#fff
    style WEAVE fill:#1abc9c,stroke:#16a085,color:#fff
    style FINAL fill:#9b59b6,stroke:#8e44ad,color:#fff
    style EVAL fill:#9b59b6,stroke:#8e44ad,color:#fff
    style ABLATION fill:#e74c3c,stroke:#c0392b,color:#fff
    style COMPARE fill:#e74c3c,stroke:#c0392b,color:#fff
    style APPLY_WINNER fill:#e74c3c,stroke:#c0392b,color:#fff
"""


def mermaid_to_image_url(mermaid_def: str) -> str:
    """Convert Mermaid definition to a renderable image URL via mermaid.ink."""
    encoded = base64.urlsafe_b64encode(mermaid_def.encode("utf-8")).decode("ascii")
    return f"https://mermaid.ink/img/{encoded}?type=png&bgColor=white"


def create_report():
    api = wandb.Api()

    flowchart_url = mermaid_to_image_url(MERMAID_DEF)

    report = Report(
        project=PROJECT,
        entity=ENTITY,
        title="Flaneur: Self-Improvement Workflow",
        description="Automated LightGCN improvement loop using Claude Code skills",
        width="readable",
        blocks=[
            H1(text="Flaneur Self-Improvement Workflow"),
            P(
                text=[
                    "This report documents the automated workflow for improving the ",
                    "LightGCN collaborative filtering model on the Gowalla dataset. ",
                    "The loop uses Claude Code skills to analyze, improve, ablate, and evaluate.",
                ]
            ),
            HorizontalRule(),
            H2(text="Workflow Flowchart"),
            Image(
                url=flowchart_url,
                caption="Flaneur self-improvement loop: analyze → improve/ablate → train → repeat → final eval",
            ),
            HorizontalRule(),
            H2(text="Skills Reference"),
            H3(text="1. /analyze-run"),
            MarkdownBlock(
                text=[
                    "**Purpose:** Query W&B for the latest training metrics and diagnose issues.\n\n",
                    "**Diagnoses:**\n",
                    "- **Underfitting** — loss still decreasing, metrics haven't plateaued\n",
                    "- **Overfitting** — train loss drops but val metrics degrade\n",
                    "- **Suboptimal hyperparameters** — converged but room for improvement\n\n",
                    "**Output:** Ranked list of recommended changes (config tweaks or code changes).",
                ]
            ),
            H3(text="2. /improve-model"),
            MarkdownBlock(
                text=[
                    "**Purpose:** Apply the top recommendation from `/analyze-run`.\n\n",
                    "**Actions:**\n",
                    "- Edits `configs/experiment/lgcn_gowalla_full.yaml` in-place\n",
                    "- May modify model code (`src/train.py`, `src/data.py`) for architectural changes\n",
                    "- Updates config comment for W&B tracking\n\n",
                    "**Rule:** One focused change at a time to isolate effects.",
                ]
            ),
            H3(text="3. /ablation"),
            MarkdownBlock(
                text=[
                    "**Purpose:** Sweep one parameter across multiple values using Hydra `--multirun`.\n\n",
                    "**Process:**\n",
                    "1. Pick parameter to sweep (embed_dim, lr, reg_weight, etc.)\n",
                    "2. Run 100-epoch sweeps with 3-4 values\n",
                    "3. Compare results in W&B\n",
                    "4. Apply the winning value to config\n\n",
                    "**Command:** `uv run python src/main.py -m experiment=lgcn_gowalla_full param=v1,v2,v3`",
                ]
            ),
            H3(text="4. /weave-eval (NEW)"),
            MarkdownBlock(
                text=[
                    "**Purpose:** Quick Weave evaluation after any training run — not just the final one.\n\n",
                    "**What it adds beyond val metrics:**\n",
                    "- **Diversity** — are recommendations varied or repetitive?\n",
                    "- **Coverage** — does the model recommend niche items or only popular ones?\n",
                    "- **Test Recall/NDCG** — held-out test set, not just validation\n\n",
                    "**Speed:** ~2 min on 100 users. No Mistral reflection overhead.\n\n",
                    "**Key insight:** The 'best' config by val recall may recommend a narrow set of popular items. "
                    "Weave eval catches this before you commit to a 1000-epoch run.",
                ]
            ),
            H3(text="5. /eval-report"),
            MarkdownBlock(
                text=[
                    "**Purpose:** Final evaluation and reporting after the improvement loop.\n\n",
                    "**Actions:**\n",
                    "- Run thorough Weave evaluation (500 users) on the best checkpoint\n",
                    "- Include Mistral reflection for qualitative analysis\n",
                    "- Generate a W&B Report summarizing the full journey\n",
                    "- Publish leaderboard for side-by-side model comparison\n\n",
                    "**When:** After the final 1000-epoch run with optimized settings.",
                ]
            ),
            HorizontalRule(),
            H2(text="Strategy"),
            MarkdownBlock(
                text=[
                    "| Phase | Epochs | Purpose |\n",
                    "|-------|--------|---------|\n",
                    "| Exploration (ablations) | 100 | Compare hyperparameter values quickly |\n",
                    "| Final run | 1000 | Squeeze out full convergence with best settings |\n\n",
                    "**Ablation order (recommended):**\n",
                    "1. `embed_dim` — capacity of the model\n",
                    "2. `reg_weight` — L2 regularization strength\n",
                    "3. `lr` — learning rate\n",
                    "4. `n_layers` — GCN depth\n",
                    "5. `n_negatives` — negative sampling ratio\n\n",
                    "Each ablation takes ~2-3 hours for 3 values. Compound winners before the final long run.",
                ]
            ),
        ],
    )

    report.save()
    print(f"Report created: {report.url}")
    return report.url


if __name__ == "__main__":
    create_report()
