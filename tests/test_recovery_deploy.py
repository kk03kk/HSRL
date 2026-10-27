import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
import zipfile

import pytest


ROOT=Path(__file__).resolve().parents[1]


def installer():
    spec=importlib.util.spec_from_file_location('native_wheel_tool',ROOT/'tools/install_native_wheel.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.install


def test_native_wheel_installs_only_into_selected_source_not_shared_environment(tmp_path):
    target=tmp_path/'release';(target/'python/taverngym').mkdir(parents=True)
    (target/'Cargo.toml').write_text('[workspace]')
    env=tmp_path/'shared-venv';env.mkdir();(env/'sentinel').write_text('unchanged')
    (target/'EXECUTION_ENV.json').write_text(json.dumps({'environment_root':str(env)}))
    wheel=tmp_path/'engine.whl'
    with zipfile.ZipFile(wheel,'w') as z:
        z.writestr('taverngym/_core.abi3.so',b'native-fixture')
        z.writestr('taverngym/rl/train.py',b'must-not-install-this')
    result=installer()(wheel,target)
    assert (target/'python/taverngym/_core.abi3.so').read_bytes()==b'native-fixture'
    assert not (target/'python/taverngym/rl/train.py').exists()
    assert (env/'sentinel').read_text()=='unchanged'
    assert json.loads((target/'EXECUTION_ENV.json').read_text())['native_sha256']==result['sha256']


def test_wheel_with_ambiguous_native_modules_is_rejected(tmp_path):
    target=tmp_path/'release';(target/'python/taverngym').mkdir(parents=True)
    (target/'Cargo.toml').write_text('[workspace]');wheel=tmp_path/'bad.whl'
    with zipfile.ZipFile(wheel,'w') as z:
        z.writestr('taverngym/_core.a.so',b'a');z.writestr('taverngym/_core.b.so',b'b')
    with pytest.raises(ValueError,match='exactly one'):installer()(wheel,target)


def test_failed_bootstrap_gate_reaches_skip_branch_under_bash_errexit():
    bash=Path('D:/git/bin/bash.exe')
    if not bash.exists():bash=shutil.which('bash')
    if not bash:pytest.skip('bash unavailable')
    source=(ROOT/'deploy/nus/recovery_train.sbatch').read_text()
    function=source[source.index('run_child() {'):source.index('if run_child python')]
    command="set -euo pipefail\n"+function+"\nif run_child bash -c 'exit 3'; then code=0; else code=$?; fi\n[ \"$code\" = 3 ] && echo GATE_REACHED\n"
    r=subprocess.run([str(bash),'-c',command],capture_output=True,text=True)
    assert r.returncode==0 and 'GATE_REACHED' in r.stdout
