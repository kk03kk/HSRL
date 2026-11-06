"""Scripted UI acceptance fixture, deliberately NOT a Battlegrounds simulator.

Snapshots are predefined checkpoints; no card rules, economy, combat or opponents.
The same Observation/Policy/Controller contracts are used by the real observer.
"""
import threading
from dataclasses import replace,asdict

from .model import Action, Card, Kind, Observation, Phase,Resolution
from .verification import verify,Status

DRAG_KINDS = (Kind.BUY, Kind.PLAY, Kind.SELL, Kind.REORDER)


def destination(action):
    if action.kind == Kind.BUY: return "hand"
    if action.kind == Kind.SELL: return "bob"
    if action.kind == Kind.REORDER: return "position:"+str(action.order.index(action.source)+1)
    if action.kind == Kind.PLAY: return f"entity:{action.target}" if action.target is not None else "board"
    if action.kind == Kind.ACTIVATE and action.target is not None:return f"entity:{action.target}"
    return None


def is_drag(action):
    return action.kind in DRAG_KINDS or (action.kind==Kind.ACTIVATE and action.target is not None)


def fixtures():
    def card(eid, attack=2, health=3):
        return Card(eid, f"TEST_{eid}", f"测试随从 {eid}", 1, "PLAY", 1, "MINION", attack, health, 1)
    a, b, c = card(101), card(102, 5, 6), card(103, 8, 8)
    rows = []
    def add(label, phase, actions=(), **fields):
        rows.append((label, Observation(revision=len(rows), game_id="scripted-fixture", mode="GT_BATTLEGROUNDS",
                    phase=phase, source="sandbox", local_player_id=1, actions=tuple(actions), **fields)))
    add("进入界面", Phase.MENU, [Action(Kind.ENTER, evidence="fixture")])
    add("选择英雄", Phase.CHOICE, [Action(Kind.CHOOSE, 101, choice_id=1), Action(Kind.CHOOSE, 102, choice_id=1)], choices=(a,b))
    add("购买随从：拖到手牌", Phase.RECRUIT, [Action(Kind.BUY, 101)], gold=3, shop=(a,))
    add("打出手牌：拖到场上", Phase.RECRUIT, [Action(Kind.PLAY, 101)], gold=0, hand=(a,))
    add("定向打牌：拖到指定目标", Phase.RECRUIT, [Action(Kind.PLAY, 102, target=101)], hand=(b,), board=(a,))
    add("分支打牌：保留分支编号",Phase.RECRUIT,[Action(Kind.PLAY,102,sub_option_id=1)],hand=(b,))
    add("专用按钮发动",Phase.RECRUIT,[Action(Kind.ACTIVATE,301)])
    add("随从定向发动",Phase.RECRUIT,[Action(Kind.ACTIVATE,102,target=101)],board=(a,b))
    add("升级酒馆", Phase.RECRUIT, [Action(Kind.UPGRADE, 201)], gold=6, board=(a,b), tavern_tier=1)
    add("刷新商店", Phase.RECRUIT, [Action(Kind.REFRESH, 202)], gold=1)
    board = tuple(card(110+i,1,1) for i in range(7))
    add("出售弱随从：拖到出售区", Phase.RECRUIT, [Action(Kind.SELL,110), Action(Kind.BUY,103)], gold=3, board=board, shop=(c,))
    add("调整站位：拖到新位置", Phase.RECRUIT, [Action(Kind.REORDER,101,order=(102,101))], board=(a,b))
    add("发现选择", Phase.CHOICE, [Action(Kind.CHOOSE,103,choice_id=2)], choices=(c,))
    add("结束操作", Phase.RECRUIT, [Action(Kind.END)])
    add("战斗等待（测试时钟推进）", Phase.COMBAT)
    add("结算界面", Phase.GAME_OVER, [Action(Kind.DISMISS)])
    add("验收完成", Phase.MENU)
    return rows


class Sandbox:
    def __init__(self, audit=None):
        self.rows = fixtures()
        self.index = 0
        self.receipts = set()
        self.stopped = False
        self.lock = threading.RLock()
        self.audit = audit
        self.resolutions=[]

    @property
    def done(self):
        return self.index == len(self.rows)-1

    def observe(self):
        with self.lock:
            return replace(self.rows[self.index][1],resolutions=tuple(self.resolutions[-128:]))

    def perform(self, action, revision):
        with self.lock:
            obs = self.observe()
            if self.stopped:
                raise RuntimeError("人工停止已触发")
            if revision != obs.revision or action not in obs.actions:
                raise ValueError("stale revision or unadvertised action")
            self.receipts.add((revision, action.key))
            self.index += 1
            self.resolutions.append(Resolution(self.index,action.kind,action.source,action.target,
                                               action.sub_option_id,action.scope,origin="sandbox_ui"))
            if self.audit:
                self.audit.write("fixture_input", revision=revision, action=action.key, next_stage=self.index)

    def acknowledged(self, action, before, after):
        outcome=verify(action,before,after)
        if self.audit:self.audit.write("postcondition",action=action.key,outcome=asdict(outcome))
        return (before.revision, action.key) in self.receipts and outcome.status in (Status.CONFIRMED,Status.RESOLVED)

    def tick(self):
        with self.lock:
            if not self.stopped and self.observe().phase == Phase.COMBAT:
                self.index += 1


class BrowserBackend:
    """Inputs go through visible DOM controls, acknowledgement through receipts."""
    def __init__(self, sandbox, page):
        self.sandbox, self.page = sandbox, page

    def observe(self):
        return self.sandbox.observe()

    def perform(self, action, revision):
        self.page.wait_for_function("r => window.fixtureRevision === r", arg=revision)
        self.page.locator("#controls").scroll_into_view_if_needed()
        control = self.page.locator(f'[data-action-key="{action.key}"]')
        if is_drag(action):
            control.hover()
            self.page.mouse.down()
            target = self.page.locator(f'[data-drop-key="{destination(action)}"]')
            target.hover()
            target.hover()  # Chromium HTML DnD needs a second move to emit dragover.
            self.page.mouse.up()
        else:
            control.click()

    def acknowledged(self, action, before, after):
        return self.sandbox.acknowledged(action,before,after)
