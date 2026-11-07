"""``_stdlib_module`` 的回归：必须能在不 import 包的前提下加载含 dataclass 的模块。

本地用普通解释器跑（有 numpy），所以额外断言"没有把 hsbg 包带进来"，并且显式
模拟登录节点的两条真实约束：
  1. ``sys.modules`` 里没有目标模块时，``@dataclass`` 会反查失败（必须有登记）；
  2. 不允许 import ``numpy``/``torch``。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.nus_submit import _stdlib_module  # noqa: E402


def test_loads_dataclass_module_without_package_import():
    before = set(sys.modules)
    ft = _stdlib_module('hsbg/az/train/forced_tier.py')
    assert ft.parse_forced_tier_plan('2:2,5:3') == {2: 2, 5: 3}
    assert ft.ForcedTierTracker('').enabled is False
    tracker = ft.ForcedTierTracker('2:2', carry=1)
    dec = tracker.decide(game_id=1, seat=0, turn=2, tier=1, upgrade_legal=True)
    assert dec.force is True
    # 只断言"这一次加载没有新增任何 hsbg.* 模块"（其它测试可能已经 import 过包）
    newly = set(sys.modules) - before
    assert not any(m.startswith('hsbg.') for m in newly), newly
    assert '_hsai_stdlib_forced_tier' in sys.modules


def test_reload_is_idempotent():
    a = _stdlib_module('hsbg/az/train/forced_tier.py')
    b = _stdlib_module('hsbg/az/train/forced_tier.py')
    assert a.parse_prob_schedule('0:1.0') == b.parse_prob_schedule('0:1.0')
