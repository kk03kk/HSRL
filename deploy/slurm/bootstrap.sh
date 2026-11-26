#!/usr/bin/env bash
#SBATCH --job-name=hsrl-bootstrap
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
run="$(realpath "${1:?provide a run directory}")"
test -d "$run"
test ! -e "$run/venv"
root="$(git rev-parse --show-toplevel)"
cd "$root"
test -z "$(git status --porcelain)"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${SLURM_CPUS_PER_TASK:?submit via Slurm}"
export CARGO_TARGET_DIR="$run/target"
export PIP_DISABLE_PIP_VERSION_CHECK=1

{
    printf 'job_id=%s\ncommit=%s\nhost=%s\nutc_started=%s\n' \
        "$SLURM_JOB_ID" "$(git rev-parse HEAD)" "$(hostname)" "$(date -u +%FT%TZ)"
    python3 --version
    rustc --version
    cargo --version
} | tee "$run/build.txt"

python3 -m venv "$run/venv"
python="$run/venv/bin/python"
"$python" -m pip install -r deploy/slurm/bootstrap-requirements.txt
"$run/venv/bin/maturin" build --locked --release -i "$python" --out "$run/wheels"
wheels=("$run"/wheels/*.whl)
test "${#wheels[@]}" -eq 1
(cd "$run" && sha256sum wheels/*.whl) | tee "$run/wheels.sha256"
"$python" -m pip install --no-deps "${wheels[0]}"
"$python" -m pip freeze > "$run/python-freeze.txt"

# 在已安装的 wheel 上验证原生扩展和完整对局。
timeout 90s "$python" - <<'PY' | tee "$run/smoke.json"
import json
from pathlib import Path

import taverngym

assert taverngym.build_is_current(Path.cwd()) is True
config = taverngym.hsrl_config_json()
games = []
for seed in range(16):
    game = taverngym.Game(seed, config)
    actions = 0
    while not game.is_over():
        before = actions
        for seat in range(game.num_players):
            if game.needs_action(seat):
                action = game.bot_action(seat)
                assert game.legal_mask(seat)[action], (seed, seat, action)
                game.step(seat, action)
                actions += 1
        assert actions > before, (seed, game.turn)

    places = list(game.placements())
    assert sorted(places) == list(range(1, 9)), (seed, places)
    assert not game.turn_limit_end, seed
    assert game.truncated_combats == game.truncated_recruit_contexts == 0, seed
    assert game.stopped_reason is None, seed
    games.append({"seed": seed, "turn": game.turn, "actions": actions, "placements": places})

print(json.dumps({
    "game_version": taverngym.game_version(),
    "rules_version": taverngym.server_rules_version(),
    "build": taverngym.build_info(),
    "games": games,
}, indent=2))
PY

# pytest 按上游配置从源码树导入，因此链接已验过的 wheel 扩展。
extension="$("$python" -c 'from taverngym import _core; print(_core.__file__)')"
# 每次作业使用各自的临时源码视图，避免并行验证修改同一冻结检出。
test_root="$run/test-source"
mkdir "$test_root"
git archive HEAD | tar -x -C "$test_root"
ln -s "$extension" "$test_root/python/taverngym/$(basename "$extension")"
(cd "$test_root" && timeout 5m "$python" -m pytest -q) | tee "$run/pytest.txt"
printf 'utc_finished=%s\n' "$(date -u +%FT%TZ)" | tee -a "$run/build.txt"
