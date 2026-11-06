"""交互阶段证明；不识别卡牌边框，不等待动画静止。"""
from .model import Kind

ORDINARY=frozenset((Kind.BUY,Kind.SELL,Kind.PLAY,Kind.UPGRADE,Kind.REFRESH,Kind.FREEZE,Kind.HERO_POWER,Kind.ACTIVATE,Kind.REORDER))


def panel_visible(obs,image):
    ctx=obs.interaction
    if ctx and ctx.display_state in ('hidden','waiting'):return False
    # 兼容旧日志的shown标签，但它不能替代当前面板出现的证据。
    from .choice_visibility import modal_hint
    return modal_hint(image)


def target_visible(obs,ui,source=None,*,parent_sent=False):
    """目标点击的上下文边界；父输入顺序或明确请求足够，不要求红箭头。"""
    if (ui or {}).get('mode') in ('pause_menu','recruit_transition','choice_transition'):return False
    ctx=obs.interaction
    if ctx:
        if ctx.mode!='target' or ctx.owner not in (None,obs.local_player_id):return False
        if source is not None and ctx.source_entity not in (None,source):return False
        return parent_sent or (ctx.protocol in ('verified_target','server_targets','fixture')
            and ctx.display_state not in ('hidden','waiting'))
    return bool(parent_sent)
