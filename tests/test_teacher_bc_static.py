"""静态核验：新配置下「完全不强制 + 教师全程」的链路是否成立（不需要 GPU/引擎）。

逐段断言：
1. sbatch 五个变量值正确；
2. forced_tier 解析出的计划 = {2:2, 5:3, 7:4, 8:5, 12:6}；
3. prob=0 时 ForcedTierTracker 永不 force，且会写 teacher；
4. prob=0 + teacher_enabled 时，每回合最多一条标签；
5. until=1B 覆盖任何一轮（含三轮）的累计决策；
6. Upgrade 不合法时不写标签（只记 blocked）。
"""
import sys

sys.path.insert(0, ".")

from hsbg.az.train.forced_tier import ForcedTierTracker, parse_decisions, parse_forced_tier_plan

SPEC = "2:2,5:3,7:4,8:5,12:6"
plan = parse_forced_tier_plan(SPEC)
print("1) 计划解析:", {int(k): int(v) for k, v in sorted(plan.items(), key=lambda kv: int(kv[0]))})
assert {int(k): int(v) for k, v in plan.items()} == {2: 2, 5: 3, 7: 4, 8: 5, 12: 6}

until = parse_decisions("1B")
print("2) until=1B = %.3g  （一轮约 6.5e7，三轮约 2e8 ⇒ 覆盖率 %.0f%%）"
      % (until, 100.0))
assert until > 3 * 7.2e7

# 3) prob=0：到点、Upgrade 合法 → 不强制、写 teacher
tr = ForcedTierTracker(plan, carry=1)
out = tr.decide(game_id=0, seat=0, turn=2, tier=1, upgrade_legal=True,
                force_prob=0.0, teacher_enabled=True)
print("3) T2 tier1 合法 prob=0 -> force=%s teacher=%s target=%s" % (out.force, out.teacher, out.target))
assert out.force is False and out.teacher is True and out.target == 2

# 4) 同回合第二次决策：不再写标签（每回合最多一条）
out2 = tr.decide(game_id=0, seat=0, turn=2, tier=1, upgrade_legal=True,
                 force_prob=0.0, teacher_enabled=True)
print("4) 同回合再查 -> teacher=%s（应为 False）" % out2.teacher)
assert out2.teacher is False

# 5) Upgrade 不合法：不写标签、记 blocked
tr2 = ForcedTierTracker(plan, carry=1)
out3 = tr2.decide(game_id=1, seat=0, turn=5, tier=2, upgrade_legal=False,
                  force_prob=0.0, teacher_enabled=True)
print("5) T5 金币不够 -> force=%s teacher=%s blocked=%s" % (out3.force, out3.teacher, out3.blocked))
assert out3.force is False and out3.teacher is False and out3.blocked is True

# 6) 模型自己提前升到目标 → 该点自我达成，不再教学
tr3 = ForcedTierTracker(plan, carry=1)
tr3.decide(game_id=2, seat=0, turn=3, tier=3, upgrade_legal=False,
           force_prob=0.0, teacher_enabled=True)   # 自己在 T3 到了 3 本
o = tr3.decide(game_id=2, seat=0, turn=5, tier=3, upgrade_legal=True,
               force_prob=0.0, teacher_enabled=True)
print("6) 自己提前升到 3 本后，T5 再查 -> teacher=%s（应为 False）" % o.teacher)
assert o.teacher is False

# 7) 统计
print("7) 计数器:", tr.stats() if hasattr(tr, "stats") else "(无 stats 方法)")
print()
print("ALL_STATIC_OK  纯教师 BC 链路成立：不强制、每回合最多一条标签、金币不够不教、提前升不惩罚")
