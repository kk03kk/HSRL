#!/bin/bash
set -euo pipefail
exec python3 ops/peer_round_series.py continue "$@"
