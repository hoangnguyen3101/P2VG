#!/bin/bash
# Quick launcher for LoRA fine-tuning — wraps scripts/train.sh.
# Usage:  bash configs/finetune_lora.sh [FOLD] [OUTPUT_SUFFIX]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"

exec bash "$P2VG_ROOT/scripts/train.sh" "$@"
