#!/bin/bash
# Wait for embed_dim sweep to finish, apply winner, then run reg_weight ablation

set -e
cd /home/ubuntu/flaneur

# Step 1: Update config comment for reg_weight ablation
# (embed_dim winner will be applied via Hydra override based on sweep results)
# For now we know 256 is winning, but we wait for final results

echo "=== Starting reg_weight ablation ==="
echo "Updating config comment..."

# Update config comment
cat > /tmp/config_update.py << 'PYEOF'
import yaml
config_path = "configs/experiment/lgcn_gowalla_full.yaml"
with open(config_path) as f:
    content = f.read()

# Replace comment lines
lines = content.split("\n")
new_lines = []
for line in lines:
    if line.startswith("#") and ("Ablation" in line or "Base" in line):
        continue
    new_lines.append(line)

header = "# @package _global_\n# Ablation: sweeping reg_weight over [1e-5,1e-4,1e-3].\n# Base: 256-dim/4-layer, lr=1e-3, n_negatives=3, 100 epochs."
new_lines[0] = header
with open(config_path, "w") as f:
    f.write("\n".join(new_lines))
PYEOF
python /tmp/config_update.py

echo "=== Running reg_weight sweep ==="
uv run python src/main.py -m experiment=lgcn_gowalla_full model.embed_dim=256 train.reg_weight=1e-5,1e-4,1e-3
echo "=== reg_weight ablation complete ==="
