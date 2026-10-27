#!/usr/bin/env bash
#SBATCH --job-name=hsrl-lib-prepare
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:45:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
python tools/package_nus.py --verify .
python -m maturin build --release --locked --out runs/library-wheel
wheel=$(find runs/library-wheel -maxdepth 1 -name '*.whl' -print -quit)
python tools/install_native_wheel.py "$wheel"
cargo fmt --all -- --check
cargo test --release --locked -p taverngym --jobs 2
python -m pytest tests -q
python -m taverngym.rl.doctor --config configs/entity_library_nus.json --device cpu
python tools/prepare_library_stage.py --config configs/entity_library_nus.json \
  --out runs/library-data-v2 --train-games 2072 --validation-games 2072 --test-games 2072 \
  --manual assets/library/manual-game1.json
printf '\nLibrary preparation complete; environment packages reused.\n'
