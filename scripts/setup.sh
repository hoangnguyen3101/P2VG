#!/bin/bash
# One-time server setup for P2VG.
# Run once after cloning / git pull on a new server.
#
# Usage:
#   bash scripts/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$P2VG_ROOT"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC}  $*"; }

echo "================================================"
echo " P2VG Setup — $(date)"
echo " Root: $P2VG_ROOT"
echo "================================================"

# ── 1. uv ─────────────────────────────────────────
echo ""
echo ">>> [1/5] Checking uv..."
if ! command -v uv &>/dev/null; then
    warn "uv not found — installing via official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to current shell session
    export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version)"

# ── 2. Python deps ────────────────────────────────
echo ""
echo ">>> [2/5] Installing Python dependencies (uv sync)..."
uv sync
ok "Python dependencies installed"

# ── 3. PYTHONPATH smoke test ──────────────────────
echo ""
echo ">>> [3/5] Smoke test imports..."
PYTHONPATH="$P2VG_ROOT/src:$P2VG_ROOT/M3D" uv run python - <<'PYEOF'
from p2vg.data.dataset import SpineCapDataset
from p2vg.model.gemma3 import LamedGemma3ForCausalLM
from p2vg.model.udml_fusion import UDMLFusion
print("Imports OK")
PYEOF
ok "Core imports work"

# ── 4. Required files / directories ───────────────
echo ""
echo ">>> [4/5] Checking required files..."

WEIGHTS_DIR="${WEIGHTS_DIR:-$P2VG_ROOT/weights}"
DATA_ROOT="${DATA_ROOT:-$P2VG_ROOT/dataset_ttd_256}"

if [ -f "$WEIGHTS_DIR/pretrained_ViT.bin" ]; then
    ok "pretrained_ViT.bin found at $WEIGHTS_DIR"
else
    warn "pretrained_ViT.bin NOT found at $WEIGHTS_DIR"
    warn "  → Place it there before running training."
fi

for split in train val test; do
    CSV="$DATA_ROOT/report/${split}.csv"
    if [ -f "$CSV" ]; then
        rows=$(tail -n +2 "$CSV" | wc -l)
        ok "${split}.csv found ($rows samples)"
    else
        warn "${split}.csv NOT found at $CSV"
        warn "  → Unzip dataset_ttd_256.zip into $P2VG_ROOT first."
    fi
done

# ── 5. PM2 (optional, for background jobs) ────────
echo ""
echo ">>> [5/5] Checking PM2 (optional)..."
if command -v pm2 &>/dev/null; then
    ok "PM2 $(pm2 --version) found"
else
    warn "PM2 not found — needed only for background training."
    warn "  Install: npm i -g pm2   (requires Node.js)"
    warn "  Or use nohup / tmux instead."
fi

# ── Summary ───────────────────────────────────────
echo ""
echo "================================================"
echo " Setup complete. Next steps:"
echo ""
echo "  # Training + auto-merge LoRA:"
echo "  bash scripts/train.sh [FOLD]"
echo ""
echo "  # Evaluate:"
echo "  bash scripts/evaluate.sh [FOLD]"
echo ""
echo "  # Background (PM2):"
echo "  pm2 start --name p2vg-train --no-autorestart -- bash scripts/train.sh 2"
echo "================================================"
