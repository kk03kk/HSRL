"""245动作实机桥接入口；默认不启用输入，旧游戏配置必须重新验收。"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'client')]


def local_player150(argv):
    """Use the installed Windows Player150 checkout and its frozen engine."""
    if sys.platform != 'win32' or argv[:1] == ['--legacy']:
        return None
    checkout = ROOT / 'artifacts/local/r1-player150-live'
    entry = checkout / 'client/live_245.py'
    if not entry.is_file():
        return None
    import json
    import os
    import subprocess

    config = json.loads((checkout/'config/live/desktop-models.json').read_text(encoding='utf8'))
    engine = checkout / config['engine']
    if not engine.is_dir():
        print('BLOCKED: missing local Player150 engine', file=sys.stderr)
        return 2
    env = dict(os.environ, PYTHONPATH=str(engine))
    return subprocess.call([sys.executable, '-X', 'utf8', str(entry), *argv], env=env)


if __name__=='__main__':
    args=sys.argv[1:]
    delegated=local_player150(args)
    if delegated is not None:
        raise SystemExit(delegated)
    if args[:1]==['--legacy']:
        args=args[1:]
    from hsrl_agent.live.cli import main
    raise SystemExit(main(argv=args,root=ROOT))
