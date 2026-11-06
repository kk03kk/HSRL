"""服务器单选请求的同步闸门：只认当前请求的完整证据，不按协议名白名单。

真实日志里同一个“单选”请求有多种客户端协议标记：

* ``server_single``            普通发现/黑暗发现（日志层即可确定）；
* ``verified_trinket_select``  饰品商店：还要四个饰品实体都已视觉映射；
* ``verified_quest_pair``      任务奖励对：还要奖励链接可解析。

旧实现只放行 ``server_single``，于是 2026-10-22 21:52 的实机局在第 6 回合
饰品商店（request_id=5, owner=4, min=max=1, display_state=shown, 4 个
BATTLEGROUND_TRINKET 选项、4 个 ``choose`` 动作全部齐备）被无限判成
“状态尚未同步”，8 秒后整局被 ``unsupported_or_stalled_interaction`` 终止。

闸门现在按**证据**判定：
1. 快照就绪且无警告；
2. 只有 ``options_pending``（本地 Options 被新日志作废、等待服务器重投）才需要
   额外的选择请求证据；
3. 证据 = 当前请求号 + 归属本地玩家 + 单选 1/1 + 未选 + 客户端已显示/未回放 +
   每个服务器 ``choose`` 动作都属于该请求且指向真实选项实体。

任何一条不满足都继续等待；不猜请求、不点未映射的牌、不把多选当单选。
"""
from bg_agent.model import Kind, Phase


def complete_single_select(obs):
    """当前快照是否是一个证据完整的服务器单选请求（不可用于多选/目标）。"""
    ctx = obs.interaction
    if obs.phase != Phase.CHOICE or ctx is None or ctx.mode != 'choice':
        return False
    if ctx.request_id is None or ctx.owner != obs.local_player_id:
        return False
    # 多选、已选、客户端尚未显示（waiting/hidden）都不能算可决策。
    if (ctx.min_select, ctx.max_select) != (1, 1) or ctx.selected:
        return False
    if ctx.display_state not in ('shown', 'unknown'):
        return False
    ids = {c.entity_id for c in obs.choices}
    if not ids:
        return False
    actions = [a for a in obs.actions if a.kind == Kind.CHOOSE]
    if not actions:
        return False
    return all(a.evidence == 'server_choices' and a.choice_id == ctx.request_id
               and a.source in ids for a in actions)


def waiting_for_sync(obs):
    """True = 只等新状态，不投递旧动作（WAIT）。"""
    if not obs.ready or obs.warnings:
        return True
    # Options属于对局操作/选择协议；已由当前UI授权的开始/结算不能等旧局Options。
    if obs.phase in (Phase.MENU,Phase.GAME_OVER):
        return False
    if not obs.options_pending:
        return False
    return not complete_single_select(obs)
