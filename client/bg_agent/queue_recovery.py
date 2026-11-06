"""Explicit failed-queue recovery, distinct from retrying a pending input."""
from functools import lru_cache
import json,time
from pathlib import Path
import cv2
import numpy as np

from .green_readiness import SignalReadiness
from .model import Action,Kind,Phase
from .native_bridge import NativePlan
from .readiness_window import capture_blockers
from .windows import Box
from .action_budgets import QUEUE_DISMISS_SECONDS


@lru_cache(maxsize=1)
def assets():
    root=Path(__file__).parent/'data/queue_error'
    profile=json.loads((root/'profile.json').read_text(encoding='utf8'))
    from PIL import Image
    templates={}
    for name in profile['regions']:
        with Image.open(root/(name+'.png')) as image:templates[name]=np.asarray(image.convert('L'))
    return profile,templates


def error_confirm(image):
    if abs(image.width/image.height-1.6)>.015:return None
    profile,templates=assets();small=image.resize(tuple(profile['size']))
    gray=np.asarray(small.convert('L'));found={}
    for name,rect in profile['regions'].items():
        x,y,right,bottom=rect;pad=6;patch=gray[y-pad:bottom+pad,x-pad:right+pad];template=templates[name]
        _,score,_,point=cv2.minMaxLoc(cv2.matchTemplate(patch,template,cv2.TM_CCOEFF_NORMED))
        px,py=point;own=patch[py:py+template.shape[0],px:px+template.shape[1]]
        if score<.91 or float(np.mean(np.abs(own.astype(float)-template)))>20:return None
        found[name]=(x-pad+px,y-pad+py,template.shape[1],template.shape[0])
    x,y,w,h=found['confirm'];hsv=cv2.cvtColor(np.asarray(small.crop((x,y,x+w,y+h))),cv2.COLOR_RGB2HSV)
    gold=(hsv[:,:,0]>=12)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>65)&(hsv[:,:,2]>100)
    if float(gold.mean())<.08:return None
    sx,sy=image.width/profile['size'][0],image.height/profile['size'][1]
    return Box(round((x+w*.25)*sx),round((y+h*.25)*sy),round(w*.5*sx),round(h*.5*sy))


def menu_context(provider,obs):
    return bool(getattr(provider,'enter_sent',False) and not getattr(provider,'matched',False)
        and obs.phase in (Phase.MENU,Phase.UNKNOWN) and obs.scene=='BACON'
        and not obs.interaction and not getattr(provider.logs,'error',None)
        and not any(w.startswith('state_health:') for w in obs.warnings))


def recover(provider,executor,before,after):
    network=getattr(getattr(provider,'lifecycle',None),'network',None)
    epoch=getattr(network,'queue_failure_epoch',0)
    if not getattr(network,'queue_cancelled',False) or not epoch or not menu_context(provider,after):return False
    row=getattr(provider,'last_row',None);image=getattr(provider,'last_image',None)
    if not row or image is None or not row.get('log_snapshot_unchanged'):return False
    if row['capture'].get('reused_capture'):return False
    if not 0<=time.time()-row['capture']['end']<=.75:return False
    if capture_blockers(row['capture']):return False
    if any(not b.startswith('network_') for b in row.get('blockers',())):return False
    ledger=getattr(provider,'checkpoint',None)
    ack=f'queue_error_ack:{epoch}';start=f'queue_restart:{epoch}'
    if epoch in getattr(provider,'recovered_queue_epochs',set()):return False
    if ledger and ledger.stage_state(start)=='sent':return False
    box=error_confirm(image)
    if box is None:return False
    prior=sum(name.startswith('queue_restart:') for name in (ledger.state.get('pending') or {}).get('stages',{})) if ledger else getattr(provider,'queue_restart_count',0)
    if prior>=3:raise RuntimeError('matchmaking_recovery_limit_reached')
    from .live_interactions import ensure_stage_unsent,record_stage,stage_click,capture_stage,archive_interaction,StageEvidenceExpired
    action=Action(Kind.ENTER,evidence='verified_client_ui')
    deadline=time.monotonic()+QUEUE_DISMISS_SECONDS
    def current():
        executor.guard()
        if executor.stopped() or executor.transport.pressed(0x77):raise InterruptedError('human_stop')
        obs=provider.semantic_observation(getattr(provider.logs,'synchronize',provider.logs.observe)())
        if str(provider.logs.session)!=str(row.get('session')):raise RuntimeError('queue_recovery_session_changed')
        window=provider.windows.snapshot(provider.binding.hwnd)
        if window.blockers or window.token!=row['window']['token']:raise RuntimeError('queue_recovery_window_changed')
        return obs,window
    def click_stage(name,obs,window,region,capture):
        ensure_stage_unsent(provider,name)
        record_stage(provider,name,'prepared',failure_epoch=epoch)
        while time.monotonic()<deadline:
            if region is not None:
                now=time.time();plan=NativePlan(obs.revision,action.key,window.token,now,now+1,0,(0,0),None,None,
                    obs.revision,obs.game_id,str(provider.logs.session))
                try:stage_click(provider,executor,action,plan,obs,window,region,stage='queue_recovery',menu_recovery=True,capture=capture)
                except StageEvidenceExpired:
                    provider.audit.write('stage_evidence_recapture',stage=name,input_sent=False)
                else:
                    record_stage(provider,name,'sent',failure_epoch=epoch)
                    return
            # 仅在输入尚未开始时重采，原准备记录和总预算均不重置。
            obs,window=current()
            if not menu_context(provider,obs):raise RuntimeError('queue_recovery_context_changed')
            frame,capture=capture_stage(provider,window,obs,action=action)
            from .live_client import menu_button
            error=error_confirm(frame)
            region=error if name==ack else menu_button(frame) if error is None else None
            if region is None:getattr(executor,'wait',executor.cancel.wait)(.05)
        raise TimeoutError('queue_recovery_stage_unconfirmed_no_retry')
    obs,window=current()
    if not menu_context(provider,obs):return False
    archive_interaction(provider,'queue-error',action,obs,window,image,row['capture'],failure_epoch=epoch)
    click_stage(ack,obs,window,box,row['capture'])
    gate=SignalReadiness()
    while time.monotonic()<deadline:
        obs,window=current()
        if not menu_context(provider,obs):return False
        frame,capture=capture_stage(provider,window,obs,action=action)
        from .live_client import menu_button
        button=menu_button(frame) if error_confirm(frame) is None else None
        proof=gate.inspect(('queue_restart',epoch),capture,
            (button.x,button.y,button.width,button.height) if button else (),valid=button is not None,window=window.token)
        if proof['ready']:
            click_stage(start,obs,window,button,capture)
            provider.recovered_queue_epochs=getattr(provider,'recovered_queue_epochs',set())|{epoch}
            provider.queue_restart_count=prior+1;provider.cache_until=0;provider.next_queue_capture=0
            provider.invalidate_readiness('explicit_queue_restarted')
            provider.audit.write('queue_error_recovered',failure_epoch=epoch,restarts=prior+1,readiness=proof)
            print('匹配被取消，已确认提示并重新匹配。',flush=True)
            return True
        getattr(executor,'wait',executor.cancel.wait)(.05)
    raise TimeoutError('queue_error_not_dismissed_no_retry')
