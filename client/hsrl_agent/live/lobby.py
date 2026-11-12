"""Bounded lobby/error state machine; consumes OCR, never sends input itself."""
from dataclasses import dataclass
import math
import re
from .contracts import Blocked


def normalized(text):return re.sub(r'[\W_]+','',text)


def trusted_rows(rows):
    return [r for r in rows if isinstance(r.get('confidence'),(int,float)) and r['confidence']>=.85
        and len(r.get('point',()))==2 and all(isinstance(x,(int,float)) and math.isfinite(x) and 0<x<1 for x in r['point'])]


SCREEN_HINTS=frozenset(('酒馆战棋','开始','开始匹配','寻找对手','正在匹配','匹配中','取消匹配','取消','点击继续','点击以继续',
    '点击屏幕继续','点击任意位置继续','点击任意位置以继续','确定','确认','发生错误','重新连接','连接失败','验证码','登录'))


#: 开始按钮的接受文本。255790 实测 OCR 为「开始」；「开始匹配」是同一控件的
#: 另一种客户端文案，接受它是严格的超集（ROI + 顶部标题两道条件不变）。
STARTS=frozenset(('开始','开始匹配'))


def screen_hint(rows):
    """True when this OCR pass recognised any calibrated lobby/error phrase.

    A false result means the half-width pass could not classify the screen, so
    the startup path pays one full-width pass before concluding anything.
    """
    return any(normalized(r.get('text','')) in SCREEN_HINTS for r in trusted_rows(rows))


RESULT_BANNER=re.compile(r'第[一二三四五六七八]名')


def result_banner(rows):
    """The Battlegrounds end-of-game banner ("第二名！"), not a generic prompt.

    255790 shows no 点击继续 text here: the banner itself is the dismiss area.
    Recognising it is what lets an unattended batch queue the next match; it is
    only ever accepted with an authoritative completed-game context.
    """
    for row in trusted_rows(rows):
        text=normalized(row.get('text',''))
        if RESULT_BANNER.search(text) and .2<row['point'][0]<.8 and .15<row['point'][1]<.85:
            return row
    return None


def queue_error(rows):
    """Observed 255790 opponent-connect failure, not a generic OK popup."""
    rows=trusted_rows(rows)
    middle=[r for r in rows if .2<r['point'][0]<.8 and .25<r['point'][1]<.72]
    text=''.join(normalized(r['text']) for r in middle)
    if not ('发生错误' in text and '对手无法连接' in text and '游戏无法继续' in text and '请再试' in text):return None
    buttons=[r for r in middle if normalized(r['text']) in ('确定','确认') and r['confidence']>=.9
             and .38<r['point'][0]<.62 and .53<r['point'][1]<.70]
    return buttons[0] if len(buttons)==1 else None


RECONNECT_TITLE='重新连接'
RECONNECT_BODY=('离线状态','连接中断')
RECONNECT_CANCEL=frozenset(('取消',))


def reconnect_dialog(rows):
    """The client's offline / "connection interrupted" dialog (255790).

    Observed 2026-11-11 06:14 (``live-e-20261111-108/match-0007``): the title and
    the confirm button share the text 重新连接, so the anchor must be the *button*
    (lower half), never the title. Attended runs must be able to recover from
    this without a human: the button is a plain client-side retry and asks for no
    credentials. 验证码/登录 still require a human and are never clicked here.

    The click is authorised only when the whole modal is positively identified:
    title above, body text present, exactly one button and exactly one 取消, both
    on the same row with the button to the LEFT of 取消. That layout condition is
    what makes it impossible to press 取消 (which closes the client) by accident.
    """
    rows=trusted_rows(rows)
    title=[r for r in rows if normalized(r['text'])==RECONNECT_TITLE and r['point'][1]<.35]
    body=[r for r in rows if any(t in normalized(r['text']) for t in RECONNECT_BODY)]
    if not title or not body:return None
    buttons=[r for r in rows if normalized(r['text'])==RECONNECT_TITLE and r['point'][1]>.6]
    cancels=[r for r in rows if normalized(r['text']) in RECONNECT_CANCEL and r['point'][1]>.6]
    if len(buttons)!=1 or len(cancels)!=1:return None
    button,cancel=buttons[0],cancels[0]
    if abs(button['point'][1]-cancel['point'][1])>.02:return None
    if not button['point'][0]<cancel['point'][0]:return None
    if button['confidence']<.9 or cancel['confidence']<.85:return None
    return button


QUEST_TITLES=frozenset(('你的任务','你的每周任务'))
#: Empty on the main menu and in the Battlegrounds lobby alike; clicking here
#: closes the quest overlay the way a player does. Kept well away from 开始/商店.
QUEST_DISMISS_POINT=(.12,.88)


def quest_popup(rows):
    """Daily/weekly quest banner shown on login and on first lobby entry.

    It has no OK button: the client closes it when you click outside the cards, so
    there is no control to re-identify. The authorisation is therefore the popup
    itself, re-checked on the fresh frame right before the click (see
    ``startup.reverify_control``), and the click always lands on a fixed empty point.
    Observed 2026-11-12 00:23 (main menu) and 00:25 (lobby); it reappears at the
    daily reset, so an unattended batch must be able to clear it.
    """
    return any(normalized(r['text']) in QUEST_TITLES for r in trusted_rows(rows))


@dataclass(frozen=True)
class LobbyAction:
    kind:str
    point:tuple


class LobbyFlow:
    def __init__(self,*,max_restarts=3):
        if type(max_restarts) is not int or not 0<=max_restarts<=10:raise Blocked('queue retries must be between 0 and 10')
        self.max_restarts=max_restarts;self.restarts=0;self.state='idle';self.proof=None
        self.sent_at=None;self.needs_restart=False;self.actions=0
        #: Client-side offline recoveries. Bounded and spaced: a reconnect that
        #: keeps failing means the network is gone, not that we should keep clicking.
        self.reconnects=0;self.max_reconnects=3
        #: Quest overlay dismissals (daily reset re-shows it mid-night).
        self.quest_dismissals=0;self.max_quest_dismissals=3
    def _reconnect_backoff(self):
        """Wait after the Nth click: 10 s, 30 s, 60 s."""
        return (10.,30.,60.)[min(max(self.reconnects-1,0),2)]
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
        banner=result_banner(rows)
        if banner:
            if not completed_gameplay:raise Blocked('result banner lacks authoritative completed-game context')
            if pending_gameplay:raise Blocked('unresolved gameplay input; result dismissal forbidden')
            if self.state=='result_pending':
                if now-self.sent_at>10:raise Blocked('result banner did not close; dismissal is not repeated')
                return None
            return self.stable('dismiss_result',banner,token,now)
        reconnect=reconnect_dialog(rows)
        if reconnect is not None:
            # Client-side offline recovery. Checked before the credential guard
            # below, because 重新连接 is in that word list and used to be refused
            # forever: on 2026-11-11 that single refusal burned ~5 unattended
            # hours and 63 match attempts.
            if self.reconnects>=self.max_reconnects:
                raise Blocked('reconnect recovery limit reached')
            if self.state=='reconnect_pending':
                if now-self.sent_at<self._reconnect_backoff():return None
            # Do not clear self.proof here: stable() keys on (kind,token) and the
            # point itself, so a proof from another control cannot authorise this
            # click. Clearing it would restart the two-frame proof on every poll
            # and the action could never be emitted.
            return self.stable('reconnect',reconnect,token,now)
        if self.state=='reconnect_pending':
            # The dialog closed on its own or after our click; re-derive the
            # lobby from a fresh frame instead of trusting the old proof.
            self.state='idle';self.needs_restart=True;self.proof=None
        if quest_popup(rows):
            # Clear the quest overlay before anything else: it covers 开始 and has no
            # dismiss button, so without this the flow waits out the whole queue
            # timeout on every match after a daily reset.
            if pending_gameplay:raise Blocked('unresolved gameplay input; quest overlay dismissal forbidden')
            if self.state=='quest_pending':
                if now-self.sent_at<8.:return None
            if self.quest_dismissals>=self.max_quest_dismissals:
                raise Blocked('quest overlay dismissal limit reached')
            return self.stable('dismiss_quests',dict(point=list(QUEST_DISMISS_POINT)),token,now)
        if self.state=='quest_pending':
            self.state='idle';self.proof=None
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
        starts=[r for r in rows if normalized(r['text']) in STARTS and .65<r['point'][0]<.95 and .65<r['point'][1]<.9]
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
        elif action.kind=='reconnect':
            self.reconnects+=1;self.state='reconnect_pending';self.needs_restart=True
        elif action.kind=='dismiss_quests':
            self.quest_dismissals+=1;self.state='quest_pending'
        else:raise Blocked('unknown lobby action')
