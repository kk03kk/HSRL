"""Execute real sbatch segment argument expansion without a GPU or trainer."""
import os,shutil,subprocess
from pathlib import Path
import pytest

def bash_binary():
    found=shutil.which('bash')
    if found:return found
    git=shutil.which('git')
    candidate=Path(git).resolve().parents[1]/'bin/bash.exe' if git else Path('/nonexistent')
    if candidate.is_file():return str(candidate)
    pytest.skip('Bash unavailable')

@pytest.mark.parametrize('resume,expected',[(0,[1,0]),(1,[0,0])])
def test_real_segment_command_preserves_loaded_anchor(tmp_path,resume,expected):
    text=Path('deploy/slurm/stage_arena.sbatch').read_text(encoding='utf-8')
    body=text[text.index('first_args=('):]
    out=tmp_path/'run';out.mkdir()
    # Replace only the executable, preserving the actual shell arrays, function,
    # continuation logic, pipes, and complete training arguments.
    script='''set -euo pipefail
out="$CAPTURE_ROOT"
ckpt=initial.pt
segments=2
resume="$CAPTURE_RESUME"
python() { printf '%s\\n' "$@" > "$out/args-$i.txt"; }
'''
    # Scientific scalar values are irrelevant to command routing, but set -u
    # should still catch genuinely misspelled script variables.
    variables=('minutes iterations stop_decisions snaps seed lr games procs epochs '
        'anchor_beta anchor_iters anchor_tail forced_tier_plan forced_tier_carry '
        'forced_tier_prob forced_teacher_until forced_teacher_coef tier_coef tier_bonus '
        'tier_table tier_gate_k tier_gate_ref tier_penalty win_bonus rank_utility_table actor_gpus').split()
    script+='\n'.join(f'{name}=0' for name in variables)+'\n'+body
    env={**os.environ,'CAPTURE_ROOT':out.as_posix(),'CAPTURE_RESUME':str(resume)}
    subprocess.run([bash_binary(),'-c',script],env=env,check=True,capture_output=True,text=True)
    for segment,reset in enumerate(expected,1):
        args=(out/f'args-{segment}.txt').read_text().splitlines()
        assert args.count('--reset-anchor')==1
        assert args[args.index('--reset-anchor')+1]==str(reset)
        assert ('--trainer-state' in args)==(resume==1 or segment>1)
