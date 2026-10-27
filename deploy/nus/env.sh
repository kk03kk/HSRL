#!/usr/bin/env bash
# Sourced by ordinary user-owned Slurm jobs. No service or custom scheduler.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_root="$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/EXECUTION_ENV.json" ]]; then
  runtime_root="$(python3 - "$PROJECT_ROOT/EXECUTION_ENV.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["environment_root"])
PY
  )" || return 1
fi
export RUSTUP_HOME="$runtime_root/.toolchains/rustup"
export CARGO_HOME="$runtime_root/.toolchains/cargo"
export PATH="$runtime_root/.venv/bin:$CARGO_HOME/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
# Source archives have deterministic (epoch) mtimes. Reusing an old target can otherwise
# keep a stale path dependency while rebuilding only the binding. Partition by contents.
native_build_id="$(python3 tools/nus_build_id.py --root "$PROJECT_ROOT")" || return 1
export CARGO_TARGET_DIR="$PROJECT_ROOT/runs/build/nus-$native_build_id"
