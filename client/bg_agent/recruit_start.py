"""每回合确认真实招募场景；对象就绪交给各动作，不等待整行像素静止。"""
import time
from .model import Phase
from .readiness_window import CaptureHistory,capture_blockers


class RecruitStartGate:
    def __init__(self,quiet=.25,max_gap=2):
        # 保留quiet构造参数兼容旧调用；它不再控制入口等待。
        self.history=CaptureHistory(max_gap)
        self.key=None;self.passed=False

    @staticmethod
    def identity(obs,session):return (str(session),obs.game_id,obs.turn)

    def allows(self,obs,session):
        return obs.phase!=Phase.RECRUIT or obs.interaction is not None or (
            self.passed and self.key==self.identity(obs,session))

    def inspect(self,row,obs,image,mappings,ui_state):
        key=self.identity(obs,row.get('session'))
        if self.key!=key:
            self.key=key;self.passed=False;self.history.invalidate()
        if self.passed:return dict(ready=True,reasons=[],already_admitted=True,turn=obs.turn,
            protocol='recruit_scene_entry',action_readiness_required=True)
        reasons=list(row.get('blockers',()))+capture_blockers(row.get('capture'))
        capture=row.get('capture') or {};stamp=capture.get('end')
        relation=self.history.inspect(('recruit_start',key),row)
        if relation.reason:reasons.append(relation.reason)
        if not isinstance(stamp,(int,float)) or not 0<=time.time()-stamp<=.75:reasons.append('capture_expired')
        if capture.get('reused_capture') or not relation.fresh:reasons.append('recruit_start_reused_capture')
        if not row.get('log_snapshot_unchanged'):reasons.append('log_capture_not_synchronized')
        if obs.phase!=Phase.RECRUIT or obs.turn is None or obs.interaction or not obs.ready:reasons.append('recruit_start_state_pending')
        # 未发布完的服务端状态不能作为新决策依据；客户端播放列表非空
        # 则交给具体对象检查，不能在入口等待所有动画完成。
        if obs.options_pending or row.get('recruit_root_processing'):
            reasons.append('recruit_start_effects_pending')
        if (ui_state or {}).get('mode') in ('recruit_transition','pause_menu','targeting_suspected'):
            reasons.append('recruit_start_overlay_visible')
        if not any(m.key=='scene:recruit' and m.status=='matched' for m in mappings):
            reasons.append('recruit_start_scene_pending')
        if reasons:
            self.history.invalidate()
            return dict(ready=False,reasons=sorted(set(reasons)),turn=obs.turn)
        # scene:recruit由当前图像的独立场景检查产生；已审核坐标或日志阶段
        # 本身不能生成它。入口通过只允许继续选动作，不是鼠标输入许可。
        self.passed=True
        return dict(ready=True,reasons=[],turn=obs.turn,already_admitted=False,
            protocol='recruit_scene_entry',capture_end=stamp,
            action_readiness_required=True,wait_for_all_animations=False)


def shop_update_pending(provider,obs,action):
    """兼容旧调用：刷新结果由动作核验负责，不再等待播放任务清空。"""
    return False
