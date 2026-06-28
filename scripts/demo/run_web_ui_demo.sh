#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

python scripts/demo/seed_demo_data.py --reset
WEB_UI_CONFIG_PATH=configs/web_ui/demo.yaml \
  python -m uvicorn web_ui.backend.main:app --host "$HOST" --port "$PORT"
