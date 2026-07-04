#!/usr/bin/env bash
# Bootstrap the development environment.
#
#   ./scripts/setup.sh          # core + dev (data/features/backtest, tests, lint)
#   ./scripts/setup.sh --ml     # also install the heavy ML stack + data providers
#
# Handy as a SessionStart hook for Claude Code on the web: it makes tests and
# linters runnable in a fresh cloud session.
set -euo pipefail

EXTRAS=(--extra dev)
if [[ "${1:-}" == "--ml" ]]; then
  EXTRAS+=(--extra ml --extra providers)
fi

uv sync "${EXTRAS[@]}"
echo "Environment ready. Try: uv run pytest"
