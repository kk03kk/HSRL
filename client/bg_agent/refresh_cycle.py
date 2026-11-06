"""刷新存在独立禁用窗口：不能借仍亮着的冻结按钮越过。"""
from .model import Kind


def refresh_cycle_signal(mapper,obs,image,mappings,action,*,pending=False):
    if action.kind!=Kind.REFRESH and not (pending and action.kind==Kind.BUY):return None
    refresh=action if action.kind==Kind.REFRESH else next((a for a in obs.actions if a.kind==Kind.REFRESH),None)
    if refresh is None:return None  # 当前不可支付/无该控件，不能借用其无效外观。
    from .green_readiness import action_signal
    signal=action_signal(mapper,obs,refresh,image,mappings)
    return {**signal,'witness_kind':'refresh','witness_entity':refresh.source,
        'reason':None if signal['ready'] else 'refresh_cycle_green_signal_absent'}
