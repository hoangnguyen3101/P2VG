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

# ── 4. Fix dataset CSV paths ──────────────────────
echo ""
echo ">>> [4/6] Fixing dataset CSV paths (absolute → relative)..."

DATA_ROOT="${DATA_ROOT:-$P2VG_ROOT/dataset_ttd_256}"

uv run python - "$DATA_ROOT" <<'PYEOF'
import csv, re, sys, pathlib

data_root = pathlib.Path(sys.argv[1])
report_dir = data_root / "report"

if not report_dir.exists():
    print(f"  report/ not found at {report_dir} — skipping (unzip dataset first)")
    sys.exit(0)

abs_pattern = re.compile(r"^.*/dataset_ttd_256/Volume")

for csv_path in sorted(report_dir.glob("*.csv")):
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8-sig")))
    if not rows:
        continue

    needs_fix = any(
        abs_pattern.match(str(row.get("image_path", ""))) or
        abs_pattern.match(str(row.get("missing_axt2_path", "")))
        for row in rows
    )
    if not needs_fix:
        print(f"  {csv_path.name}: already clean")
        continue

    for row in rows:
        for col in ("image_path", "missing_axt2_path"):
            if col in row and row[col]:
                row[col] = abs_pattern.sub("Volume", row[col])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {csv_path.name}: fixed")
PYEOF
ok "CSV paths checked"

# ── 5. Required files / directories ───────────────
echo ""
echo ">>> [5/6] Checking required files..."

WEIGHTS_DIR="${WEIGHTS_DIR:-$P2VG_ROOT/weights}"

mkdir -p "$WEIGHTS_DIR"
if [ -f "$WEIGHTS_DIR/pretrained_ViT.bin" ]; then
    ok "pretrained_ViT.bin found at $WEIGHTS_DIR"
else
    warn "pretrained_ViT.bin NOT found — downloading from HuggingFace..."
    if command -v hf &>/dev/null; then
        hf download GoodBaiBai88/M3D-CLIP pretrained_ViT.bin --local-dir "$WEIGHTS_DIR"
        ok "pretrained_ViT.bin downloaded"
    else
        warn "  → 'hf' not found. Install: uv tool install hf"
        warn "  → Then run: hf download GoodBaiBai88/M3D-CLIP pretrained_ViT.bin --local-dir $WEIGHTS_DIR"
    fi
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

# ── 6. PM2 (optional, for background jobs) ────────
echo ""
echo ">>> [6/6] Checking PM2 (optional)..."
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
