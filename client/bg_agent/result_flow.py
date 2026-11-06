"""Distinct result pages are separate one-shot stages, never click retries."""
import time
import cv2
import numpy as np

from .green_readiness import SignalReadiness
from .model import Kind,Phase
from .windows import Box


def rewards_page(image):
    """Grey battle backdrop plus independent blue XP and pink rating bars."""
    small=image.resize((800,500));hsv=cv2.cvtColor(np.asarray(small),cv2.COLOR_RGB2HSV)
    if float(np.mean(hsv[:400,:,1]<65))<.6:return None
    bands=[]
    for left,right,h0,h1,span,ratio in ((.14,.46,95,130,.06,1.8),(.58,.91,140,175,.12,2.5)):
        x0,x1=round(800*left),round(800*right);y0,y1=round(500*.83),round(500*.95)
        patch=hsv[y0:y1,x0:x1];h,s,v=cv2.split(patch)
        mask=((h>=h0)&(h<=h1)&(s>90)&(v>100)).astype('uint8')*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,5),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        spans=[(x,y,w,hh) for c in contours for x,y,w,hh in [cv2.boundingRect(c)]
               if w>800*span and 3<hh<500*.1 and w/hh>ratio]
        if not spans:return None
        bands.append(max(spans,key=lambda b:b[2]))
    # This page accepts a background click. The source is inside the observed
    # result overlay, clear of the two progress bars and settings corner.
    return Box(round(image.width*.49),round(image.height*.49),max(6,round(image.width*.02)),max(6,round(image.height*.02)))


def finish_rewards(provider,executor,action,plan):
    if action.kind!=Kind.DISMISS or provider.last_obs.phase!=Phase.GAME_OVER:return
    from .live_interactions import ensure_stage_unsent,capture_stage,record_stage,stage_click,archive_interaction,StageEvidenceExpired
    ensure_stage_unsent(provider,'result_rewards')
    # The saved reward-entry animation is a different page from the final
    # bars, still present at the old 12-second verification deadline. Poll
    # for the actual next page; this is an upper bound, never a fixed sleep.
    before=provider.last_obs;deadline=time.monotonic()+20;readiness=SignalReadiness();first=True;prepared=False
    while time.monotonic()<deadline:
        executor.guard()
        if executor.stopped() or executor.transport.pressed(0x77):raise InterruptedError('human_stop')
        obs=provider.semantic_observation(getattr(provider.logs,'synchronize',provider.logs.observe)())
        if obs.phase==Phase.MENU:return
        window=provider.windows.snapshot(provider.binding.hwnd)
        if window.token!=plan.window_token or window.blockers:raise RuntimeError('result_window_changed')
        if (obs.game_id,obs.local_player_id,obs.mode)!=(before.game_id,before.local_player_id,before.mode) or obs.phase!=Phase.GAME_OVER:
            raise RuntimeError('result_context_changed')
        image,capture=capture_stage(provider,window,obs,action=action)
        box=rewards_page(image)
        evidence=readiness.inspect(('result_rewards',before.game_id),capture,
            (box.x,box.y,box.width,box.height) if box else (),
            valid=box is not None and obs.ready and not getattr(provider.logs,'error',None),window=window.token)
        if first:
            archive_interaction(provider,'result-page-first',action,obs,window,image,capture,readiness=evidence);first=False
        if evidence['ready']:
            archive_interaction(provider,'result-rewards-ready',action,obs,window,image,capture,readiness=evidence)
            if not prepared:
                record_stage(provider,'result_rewards','prepared',game=obs.game_id);prepared=True
            try:stage_click(provider,executor,action,plan,obs,window,box,stage='result_rewards',capture=capture)
            except StageEvidenceExpired:
                provider.audit.write('stage_evidence_recapture',stage='result_rewards',input_sent=False)
                readiness.invalidate();continue
            record_stage(provider,'result_rewards','sent',game=obs.game_id)
            provider.audit.write('result_rewards_dismissed',game_id=obs.game_id,readiness=evidence,retry=False)
            return
        wait=getattr(executor,'wait',executor.cancel.wait);wait(.05)
    # Unknown pages cannot authorize another click. Normal result verification
    # remains responsible for reporting failure to return to the menu.
    provider.audit.write('result_followup_unconfirmed',game_id=before.game_id,additional_input_sent=False)
