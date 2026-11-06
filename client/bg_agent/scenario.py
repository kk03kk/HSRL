"""Multi-round interaction contract fixture, NOT a combat/economy simulator.

Stages and effects are deliberately authored, not inferred from Hearthstone rules.
Real DOM input selects advertised actions; animations/combat advance on the clock.
"""
from dataclasses import dataclass, replace
import threading
import time

from .execution import Delivery, Dispatch
from .model import Action, Card, Kind, Observation, Phase, Resolution


@dataclass(frozen=True)
class Stage:
    label: str
    observation: Observation
    duration: float | None = None


def match_stages():
    stages = []
    obs = Observation(game_id="v03-contract-match", mode="GT_BATTLEGROUNDS", source="sandbox",
                      local_player_id=1, phase=Phase.MENU, gold=3, tavern_tier=1)

    def card(eid, atk=2, hp=3, **kw):
        return Card(eid, str(eid), f"受控卡牌 {eid}", 1, "PLAY", 1, "MINION", atk, hp, 1, **kw)

    def stage(label, actions=(), *, duration=None, **fields):
        nonlocal obs
        obs=replace(obs, revision=len(stages), actions=tuple(actions), **fields)
        stages.append(Stage(label, obs, duration))

    a,b,c=card(101),card(102,5,6),card(103,8,8)
    h1,h2=card(1,0,30),card(2,1,30)
    stage("入口", [Action(Kind.ENTER)])
    stage("英雄单选", [Action(Kind.CHOOSE,2,choice_id=1),Action(Kind.CHOOSE,1,choice_id=1)], phase=Phase.CHOICE, choices=(h1,h2))
    stage("第一轮购买", [Action(Kind.BUY,101)], phase=Phase.RECRUIT, turn=1, choices=(),hero=h2,shop=(a,))
    stage("第一轮打牌", [Action(Kind.PLAY,101)], gold=0,shop=(),hand=(replace(a,zone="HAND"),))
    stage("购买第二张随从", [Action(Kind.BUY,102)], board=(a,),hand=(),gold=3,shop=(b,))
    stage("战吼选目标", [Action(Kind.PLAY,102,target=101)],shop=(),gold=0,hand=(replace(b,zone="HAND"),))
    stage("英雄技能选目标", [Action(Kind.HERO_POWER,900,target=102)],hand=(),board=(a,b))
    stage("酒馆升本", [Action(Kind.UPGRADE,901)],gold=4)
    stage("第一轮站位", [Action(Kind.REORDER,102,order=(102,101),scope="board")],tavern_tier=2,gold=0)
    stage("第一轮结束", [Action(Kind.END)],board=(replace(b,position=1),replace(a,position=2)))
    stage("第一轮战斗",phase=Phase.COMBAT,duration=.45)
    stage("第二轮发现",[Action(Kind.CHOOSE,103,choice_id=2),Action(Kind.CHOOSE,104,choice_id=2)],phase=Phase.CHOICE,turn=2,choices=(c,card(104)),gold=6)
    stage("发现奖励打出",[Action(Kind.PLAY,103)],phase=Phase.RECRUIT,choices=(),hand=(replace(c,zone="HAND"),))
    spell=replace(card(401,0,0),card_type="BATTLEGROUND_SPELL",name="受控定向法术")
    stage("购买酒馆法术",[Action(Kind.BUY,401)],hand=(),shop=(spell,),board=(b,a,c))
    stage("酒馆法术选目标",[Action(Kind.PLAY,401,target=103)],shop=(),hand=(replace(spell,zone="HAND"),),gold=4)
    stage("带分支的打牌",[Action(Kind.PLAY,402,sub_option_id=1,target=103)],hand=(replace(spell,entity_id=402,zone="HAND"),),board=(b,a,replace(c,attack=9)))
    stage("专用控件发动",[Action(Kind.ACTIVATE,903)],hand=())
    stage("随从定向发动",[Action(Kind.ACTIVATE,103,target=101)],board=(b,a,c))
    full=tuple(replace(card(200+i,1,1),position=i+1) for i in range(7))
    stage("满场先出售",[Action(Kind.SELL,200)],board=full,hand=(replace(c,zone="HAND"),))
    stage("空位打牌",[Action(Kind.PLAY,103,position=7)],board=full[1:],gold=5)
    hand=tuple(replace(card(500+i),zone="HAND") for i in range(10))
    stage("满手只允许先打牌",[Action(Kind.PLAY,500,position=1)],board=(),hand=hand)
    stage("满手解除后购买",[Action(Kind.BUY,600)],board=(card(500),),hand=hand[1:],shop=(card(600),))
    stage("第二轮刷新",[Action(Kind.REFRESH,902)],hand=(*hand[1:],replace(card(600),zone="HAND")),shop=(card(601),),gold=1)
    stage("第二轮结束",[Action(Kind.END)],shop=(card(602),),gold=0)
    stage("第二轮战斗",phase=Phase.COMBAT,duration=.45)
    dup1=replace(card(700),card_id="TRIPLE_BASE")
    dup2=replace(card(701),card_id="TRIPLE_BASE")
    dup3=replace(card(702),card_id="TRIPLE_BASE")
    golden=replace(card(703,4,6),card_id="TRIPLE_GOLDEN",tags={"PREMIUM":"1"})
    stage("第三轮三连购买",[Action(Kind.BUY,702)],phase=Phase.RECRUIT,turn=3,board=(dup1,dup2),hand=(),shop=(dup3,),gold=3)
    stage("三连合成后打出金色随从",[Action(Kind.PLAY,703)],board=(),shop=(),hand=(replace(golden,zone="HAND"),),gold=0)
    stage("三连强制发现",[Action(Kind.CHOOSE,800,choice_id=3),Action(Kind.CHOOSE,801,choice_id=3)],phase=Phase.CHOICE,board=(golden,),hand=(),choices=(card(800,12,12),card(801)))
    stage("三连奖励打出",[Action(Kind.PLAY,800)],phase=Phase.RECRUIT,choices=(),hand=(replace(card(800,12,12),zone="HAND"),))
    stage("第三轮结束",[Action(Kind.END)],hand=(),board=(golden,card(800,12,12)))
    stage("第三轮战斗",phase=Phase.COMBAT,duration=.45)
    stage("结算并返回",[Action(Kind.DISMISS)],phase=Phase.GAME_OVER)
    stage("完成",phase=Phase.MENU)
    return stages


class Scenario:
    def __init__(self, audit=None, *, faults=None, delay=.09, stages=None):
        self.rows=stages or match_stages()
        self.index=0
        self.audit=audit
        self.lock=threading.RLock()
        self.stopped=False
        self.resolutions=[]
        self.faults=dict(faults or {})  # Label -> reject_once/drop/ambiguous/temporary_missing
        self.injected=set()
        self.delay=delay
        self.pending=None
        self.entered=time.monotonic()
        self.accepted=0

    @property
    def done(self):
        return self.index==len(self.rows)-1

    @property
    def label(self):
        return self.rows[self.index].label

    def _advance(self, action=None):
        self.index+=1
        self.entered=time.monotonic()
        if action:
            self.resolutions.append(Resolution(self.index,action.kind,action.source,action.target,
                                               action.sub_option_id,action.scope,origin="sandbox_ui"))

    def observe(self):
        with self.lock:
            if self.pending and time.monotonic()>=self.pending[0]:
                _,action=self.pending
                self.pending=None
                self._advance(action)
            stage=self.rows[self.index]
            if not self.stopped and stage.duration and time.monotonic()-self.entered>=stage.duration:
                self._advance()
                stage=self.rows[self.index]
            obs=replace(stage.observation,captured_at=time.time(),source_updated_at=time.time(),
                        ready=self.pending is None,resolutions=tuple(self.resolutions))
            if self.faults.get(stage.label)=="temporary_missing" and time.monotonic()-self.entered<.18:
                return replace(obs,ready=False,hand=(),board=(),actions=(),warnings=("animation_partial_snapshot",))
            return obs

    def submit(self, action, revision):
        with self.lock:
            obs=self.observe()
            if self.stopped or not obs.ready or revision!=obs.revision or action not in obs.actions:
                return Dispatch(Delivery.REJECTED,"stopped, busy or stale; no effect applied")
            fault=self.faults.get(self.label)
            if fault=="reject_once" and self.label not in self.injected:
                self.injected.add(self.label)
                if self.audit:self.audit.write("fault_injected",label=self.label,fault=fault)
                return Dispatch(Delivery.REJECTED,"controlled explicit rejection before any effect")
            self.accepted+=1
            if self.audit:self.audit.write("fixture_input",label=self.label,revision=revision,action=action.key,fault=fault)
            if fault=="drop":
                return Dispatch(Delivery.ACCEPTED,"input delivered but effect intentionally missing")
            self.pending=(time.monotonic()+self.delay,action)
            if fault=="ambiguous":
                return Dispatch(Delivery.AMBIGUOUS,"receiver applied input but transport acknowledgement lost")
            return Dispatch(Delivery.ACCEPTED,"controlled UI input accepted; await stable state")

    def perform(self, action, revision):
        return self.submit(action, revision)

    def execute(self, action, revision):
        return self.submit(action, revision)
