"""无绿光来源和刚重排区域的局部就绪证据，不等待整屏动画。"""
import time
import numpy as np
from .action_context import input_context_token
from .action_geometry import route
from .readiness_window import CaptureHistory, capture_blockers


class LocalActionReady:
    def __init__(self):
        self.history = CaptureHistory()
        self.previous = None

    def invalidate(self):
        self.history.invalidate()
        self.previous=None

    def inspect(self, row, obs, action, image, mappings, mapper, *, idle_required=False):
        reasons = list(row.get('blockers', ())) + capture_blockers(row.get('capture'))
        stamp = (row.get('capture') or {}).get('end')
        if not isinstance(stamp, (int, float)) or not 0 <= time.time()-stamp <= .75:
            reasons.append('capture_expired')
        if stamp is not None and stamp <= row.get('input_released_at', float('-inf')):
            reasons.append('capture_precedes_input_release')
        if not row.get('log_snapshot_unchanged') or not obs.ready or obs.options_pending:
            reasons.append('log_capture_not_synchronized')
        choice_panel=str(obs.phase)=='choice' and str(action.kind)=='choose'
        if (obs.interaction and not choice_panel) or (idle_required and (obs.client_task_list is not None or row.get('recruit_root_processing'))):
            reasons.append('local_action_processing')
        if row.get('capture', {}).get('reused_capture'):
            reasons.append('local_capture_reused')
        visible = {m.key:m for m in mappings if m.status == 'matched' and m.rect}
        geometry = route(action, obs, visible)
        if geometry.error or any(k not in visible for k in geometry.required):
            reasons.append('action_source_geometry_unconfirmed')
        elif not getattr(mapper, 'authorize_geometry', lambda *a:False)(obs, action):
            # 非校准路径仍须使用原逐对象几何检查，不能用灰色背景代替来源绑定。
            from .green_readiness import source_geometry_ready
            if not source_geometry_ready(mapper, obs, action, mappings):
                reasons.append('action_source_geometry_unconfirmed')
        context = (input_context_token(obs, action), action.key,row.get('input_released_at',0.))
        relation = self.history.inspect(context, row)
        if relation.reason: reasons.append(relation.reason)
        crops = []
        if not reasons:
            for key in geometry.required:
                box = visible[key].rect
                # 包含来源边缘和相邻接收区域，尺寸由当前映射决定。
                px, py = max(8, round(image.width*.025)), max(8, round(image.height*.04))
                bounds = (max(0,box.x-px),max(0,box.y-py),
                    min(image.width,box.x+box.width+px),min(image.height,box.y+box.height+py))
                crops.append(np.asarray(image.crop(bounds).resize((80,80)), dtype=np.float32))
        prior = self.previous
        motion=getattr(mapper,'motion_evidence',lambda *a:None)(obs,action,image) if not reasons else None
        if motion is False:reasons.append('action_source_geometry_unconfirmed')
        quiet=.15 if choice_panel else .033
        stable = bool(not reasons and relation.comparable and relation.fresh and prior
            and prior[0] == context and len(prior[2]) == len(crops) and stamp-prior[1] >= quiet)
        if stable:
            if motion is not None and prior[3] is not None:
                stable &= len(motion)==len(prior[3]) and max((abs(a-b) for a,b in zip(motion,prior[3])),default=0)<=4
            # 已验证的实际轮廓位置优先；卡面呼吸/入场光效不能延长局部等待。
            if motion is None:
                for old, new in zip(prior[2], crops):
                    delta = np.abs(old-new)
                    stable &= float(delta.mean()) <= 3 and float((delta.max(axis=2)>18).mean()) <= .04
        if reasons:
            self.previous = None
            self.history.invalidate()
        elif not prior or prior[0] != context or stamp-prior[1] >= quiet:
            self.previous = (context, stamp, crops,motion)
        return dict(ready=bool(stable), reasons=sorted(set(reasons)) if reasons else
            ([] if stable else ['local_object_picture_pending']), checked_at=stamp,
            protocol='no_green_source_local_geometry' if idle_required else 'changed_row_local_geometry')
