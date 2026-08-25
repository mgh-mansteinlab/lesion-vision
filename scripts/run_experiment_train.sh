#!/usr/bin/env bash
# Launch training from configs/experiments/*.yaml (see experiments/README.md).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
YAML="${1:?Usage: $0 configs/experiments/<name>.yaml}"
cd "${PROJECT_DIR}"
exec python3 "${SCRIPT_DIR}/launch_experiment.py" "${YAML}"
