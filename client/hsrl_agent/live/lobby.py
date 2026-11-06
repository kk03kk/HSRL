"""Bounded lobby/error state machine; consumes OCR, never sends input itself."""
from dataclasses import dataclass
import math
import re
from .contracts import Blocked


def normalized(text):return re.sub(r'[\W_]+','',text)


def trusted_rows(rows):
    return [r for r in rows if isinstance(r.get('confidence'),(int,float)) and r['confidence']>=.85
        and len(r.get('point',()))==2 and all(isinstance(x,(int,float)) and math.isfinite(x) and 0<x<1 for x in r['point'])]


SCREEN_HINTS=frozenset(('酒馆战棋','开始','寻找对手','正在匹配','匹配中','取消匹配','取消','点击继续','点击以继续',
    '点击屏幕继续','点击任意位置继续','点击任意位置以继续','确定','确认','发生错误','重新连接','连接失败','验证码','登录'))


def screen_hint(rows):
    """True when this OCR pass recognised any calibrated lobby/error phrase.

    A false result means the half-width pass could not classify the screen, so
    the startup path pays one full-width pass before concluding anything.
    """
    return any(normalized(r.get('text','')) in SCREEN_HINTS for r in trusted_rows(rows))


def queue_error(rows):
    """Observed 255790 opponent-connect failure, not a generic OK popup."""
    rows=trusted_rows(rows)
    middle=[r for r in rows if .2<r['point'][0]<.8 and .25<r['point'][1]<.72]
    text=''.join(normalized(r['text']) for r in middle)
    if not ('发生错误' in text and '对手无法连接' in text and '游戏无法继续' in text and '请再试' in text):return None
    buttons=[r for r in middle if normalized(r['text']) in ('确定','确认') and r['confidence']>=.9
             and .38<r['point'][0]<.62 and .53<r['point'][1]<.70]
    return buttons[0] if len(buttons)==1 else None


@dataclass(frozen=True)
class LobbyAction:
    kind:str
    point:tuple


class LobbyFlow:
    def __init__(self,*,max_restarts=3):
        if type(max_restarts) is not int or not 0<=max_restarts<=10:raise Blocked('queue retries must be between 0 and 10')
        self.max_restarts=max_restarts;self.restarts=0;self.state='idle';self.proof=None
        self.sent_at=None;self.needs_restart=False;self.actions=0
    def stable(self,kind,row,token,now):
        old=self.proof;point=tuple(row['point']);key=(kind,token)
        if old and old[0]==key and .15<=now-old[2]<=3 and max(abs(a-b) for a,b in zip(point,old[1]))<=.004:
            return LobbyAction(kind,point)
        if old and old[0]==key and now-old[2]<.15 and max(abs(a-b) for a,b in zip(point,old[1]))<=.004:
            return None
        self.proof=(key,point,now);return None
    def inspect(self,rows,token,now,*,pending_gameplay=False,completed_gameplay=False):
        rows=trusted_rows(rows);error=queue_error(rows)
        if error:
            if not any(normalized(r['text'])=='酒馆战棋' and r['point'][1]<.12 for r in rows):
                raise Blocked('matching error is outside verified Battlegrounds lobby')
            if pending_gameplay:raise Blocked('unresolved gameplay input; matchmaking restart forbidden')
            if self.state=='dismiss_pending':
                if now-self.sent_at>6:raise Blocked('matching error did not close; confirm is not repeated')
                return None
            if self.restarts>=self.max_restarts:raise Blocked('matchmaking recovery limit reached')
            self.state='error_proof'
            return self.stable('dismiss_error',error,token,now)
        texts=[normalized(r['text']) for r in rows]
        result_prompts={'点击继续','点击以继续','点击屏幕继续','点击任意位置继续','点击任意位置以继续'}
        prompts=[r for r in rows if normalized(r['text']) in result_prompts and .2<r['point'][0]<.8 and .5<r['point'][1]<.96]
        if prompts:
            if not completed_gameplay:raise Blocked('result prompt lacks authoritative completed-game context')
            if pending_gameplay:raise Blocked('unresolved gameplay input; result dismissal forbidden')
            if self.state=='result_pending':
                if now-self.sent_at>10:raise Blocked('result did not close; dismissal is not repeated')
                return None
            if len(prompts)!=1:raise Blocked('result prompt is ambiguous')
            return self.stable('dismiss_result',prompts[0],token,now)
        if any(t in ('发生错误','确定','确认','重新连接','连接失败','验证码','登录') for t in texts):
            if self.state=='dismiss_pending' and now-self.sent_at<=6 and not any(t in ('验证码','登录') for t in texts):
                # Closing animations can temporarily hide only the body text.
                # Wait without clicking; never reinterpret a partial modal as a
                # clear lobby or send another confirm.
                self.proof=None;return None
            raise Blocked('unrecognized lobby popup or login/connection intervention needed')
        searching=any(any(word in t for word in ('寻找对手','正在匹配','匹配中','取消匹配')) or t=='取消' for t in texts)
        if searching:return None
        header=any(normalized(r['text'])=='酒馆战棋' and r['point'][1]<.12 for r in rows)
        starts=[r for r in rows if normalized(r['text'])=='开始' and .65<r['point'][0]<.95 and .65<r['point'][1]<.9]
        start=starts[0] if header and len(starts)==1 else None
        if self.state=='error_proof':
            # The user/client may already have dismissed the positively seen
            # failure. A fresh two-frame lobby proof is still required.
            self.needs_restart=True;self.state='dismiss_pending';self.sent_at=now;self.proof=None
        if self.state=='dismiss_pending':
            if start:
                if pending_gameplay:raise Blocked('unresolved gameplay input; matchmaking restart forbidden')
                return self.stable('start',start,token,now)
            if now-self.sent_at>10:raise Blocked('lobby not verified after matching error dismissal')
            return None
        if self.state=='awaiting_match':return None  # Never toggle a pending queue.
        if start:
            if pending_gameplay:raise Blocked('unresolved gameplay input; new matchmaking forbidden')
            return self.stable('start',start,token,now)
        if self.state=='waiting_lobby':return None
        if self.state=='result_pending':
            if now-self.sent_at>10:raise Blocked('lobby not verified after result dismissal')
            return None
        entries=[r for r in rows if normalized(r['text'])=='酒馆战棋' and r['point'][1]>.15]
        if len(entries)==1:return self.stable('enter_lobby',entries[0],token,now)
        return None
    def commit(self,action,now):
        """Record intent before the first button down; failures stop this run."""
        self.actions+=1;self.proof=None;self.sent_at=now
        if action.kind=='dismiss_error':self.state='dismiss_pending';self.needs_restart=True
        elif action.kind=='start':
            if self.needs_restart:self.restarts+=1
            self.needs_restart=False;self.state='awaiting_match'
        elif action.kind=='enter_lobby':self.state='waiting_lobby'
        elif action.kind=='dismiss_result':self.state='result_pending'
        else:raise Blocked('unknown lobby action')
