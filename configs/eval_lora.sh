#!/bin/bash
# Quick launcher for evaluation — wraps scripts/evaluate.sh.
# Usage:  bash configs/eval_lora.sh [FOLD] [MODEL_SUBDIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"

exec bash "$P2VG_ROOT/scripts/evaluate.sh" "$@"
