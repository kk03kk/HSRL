"""245动作实机桥接入口；默认不启用输入，旧游戏配置必须重新验收。"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'client')]
if __name__=='__main__':
    from hsrl_agent.live.cli import main
    raise SystemExit(main(root=ROOT))
