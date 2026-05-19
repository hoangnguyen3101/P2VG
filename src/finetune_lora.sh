#!/bin/bash
# Compatibility launcher. The branch defaults live in scripts/train.sh:
# fused sagittal, no axial encoder, no UDML.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"

exec bash "$P2VG_ROOT/scripts/train.sh" "$@"
