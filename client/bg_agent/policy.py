"""Small deterministic baseline. Accepts only actions advertised by an adapter."""
from dataclasses import dataclass,replace
from collections import Counter
import time
from .model import Action, Decision, Kind, Observation, Phase


@dataclass(frozen=True)
class RuleProfile:
    opening: bool = True
    dark_gifts_per_turn: int = 1
    coverage: bool = True


class CoverageRules:
    """Optional acceptance policy. Counts committed outcomes, never decisions."""
    def __init__(self,profile):
        self.profile=profile;self.done=Counter();self.frozen_turn={};self.opening_waits={}

    def opening_wait(self,obs,kind):
        """Reserve a short grace period, never an entire turn for missing options."""
        key=(obs.game_id,obs.turn,kind)
        now=obs.captured_at or time.time()
        first=self.opening_waits.setdefault(key,now)
        return 0<=now-first<2

    @staticmethod
    def gift(obs,a):
        return a.kind==Kind.ACTIVATE and any(c.entity_id==a.source and c.card_id=='BG36_Button_DarkGift' for c in obs.controls)

    def count(self,obs,kind,turn=None,source=None):
        return self.done[(obs.game_id,obs.turn if turn is None else turn,kind,source)]

    def verified(self,a,before,after):
        kind='dark_gift' if self.gift(before,a) else str(a.kind)
        self.done[(before.game_id,before.turn,kind,None)]+=1
        if a.source is not None:self.done[(before.game_id,before.turn,kind,a.source)]+=1
        self.done[(before.game_id,'all',kind,None)]+=1
        if a.kind==Kind.ACTIVATE and kind!='dark_gift':
            card=next((c for c in before.board+before.controls if c.entity_id==a.source),None)
            if card:self.done[(before.game_id,'all','activate_card',card.card_id)]+=1
        if a.kind==Kind.FREEZE and after.shop and all(c.tags.get('FROZEN')=='1' for c in after.shop):
            self.frozen_turn[before.game_id]=before.turn

    def filter(self,obs):
        actions=[]
        for a in obs.actions:
            if self.gift(obs,a):
                if self.profile.opening and obs.turn in (3,4,5) and self.count(obs,'dark_gift')<self.profile.dark_gifts_per_turn:actions.append(a)
                continue
            if self.profile.coverage and a.kind in (Kind.HERO_POWER,Kind.ACTIVATE) and self.count(obs,str(a.kind),source=a.source):continue
            if self.profile.coverage and a.kind==Kind.ACTIVATE:
                card=next((c for c in obs.board+obs.controls if c.entity_id==a.source),None)
                if card and self.count(obs,'activate_card',turn='all',source=card.card_id):continue
            actions.append(a)
        return replace(obs,actions=tuple(actions))

    def priority(self,obs,score):
        def first(kind):return RulePolicy.first(obs,kind)
        def answer(a,why):return Decision(a or Action(Kind.WAIT),obs.revision,why)
        if obs.phase!=Phase.RECRUIT:return None
        if self.profile.opening:
            if obs.turn==1 and not self.count(obs,'buy'):
                cards={c.entity_id:c for c in obs.shop if c.card_type=='MINION'}
                buys=[a for a in obs.actions if a.kind==Kind.BUY and a.source in cards]
                if buys or self.opening_wait(obs,'buy'):
                    return answer(max(buys,key=lambda a:score(cards[a.source])) if buys else None,'固定开局：第1回合购买随从，短暂等待服务器合法购买动作')
            if obs.turn==2 and not self.count(obs,'upgrade') and (obs.tavern_tier or 1)<2:
                if first(Kind.UPGRADE) or self.opening_wait(obs,'upgrade'):
                    return answer(first(Kind.UPGRADE),'固定开局：第2回合优先升至2本；合法选项短暂缺失时等待')
            if obs.turn in (3,4,5) and self.count(obs,'dark_gift')<self.profile.dark_gifts_per_turn:
                gift=next((a for a in obs.actions if self.gift(obs,a)),None)
                if gift:return answer(gift,'固定开局：第3至5回合优先黑暗之赐，每次重新核验')
                # Existing cards can generate the resources/choice needed for
                # the scheduled gift. Never spend its budget on a reroll.
                if first(Kind.PLAY) or self.opening_wait(obs,'gift'):
                    return answer(first(Kind.PLAY),'固定开局：短暂等待黑暗之赐；超时后继续合法招募')
        if not self.profile.coverage:return None
        if a:=first(Kind.PLAY):return answer(a,'优先打出手牌并完成其强制交互')
        frozen=bool(obs.shop) and all(c.tags.get('FROZEN')=='1' for c in obs.shop)
        if frozen and obs.turn is not None and obs.turn>self.frozen_turn.get(obs.game_id,obs.turn):
            if a:=first(Kind.FREEZE):return answer(a,'上一回合保留的商店已检查，解锁后继续招募')
        for kind in (Kind.HERO_POWER,Kind.ACTIVATE):
            if a:=first(kind):return answer(a,'覆盖当前合法主动能力；同一来源每回合最多一次')
        if not self.count(obs,'sell',turn='all') and len(obs.board)>=3 and (obs.gold or 0)==2:
            cards={c.entity_id:c for c in obs.shop if c.card_type=='MINION'}
            weakest=min(obs.board,key=score)
            if any(score(c)>score(weakest)+2 for c in cards.values()):
                a=next((a for a in obs.actions if a.kind==Kind.SELL and a.source==weakest.entity_id),None)
                if a:return answer(a,'出售最弱随从，补足金币以替换明显更强的商店随从')
        if not self.count(obs,'buy_spell',turn='all') and obs.board:
            spells={c.entity_id for c in obs.shop if c.card_type in ('SPELL','BATTLEGROUND_SPELL')}
            a=next((a for a in obs.actions if a.kind==Kind.BUY and a.source in spells),None)
            if a:return answer(a,'覆盖一次合法酒馆法术购买，随后按合法目标打出')
        return None


class RulePolicy:
    def __init__(self,profile=None):self.coverage=CoverageRules(profile) if profile else None

    def on_verified(self,action,before,after,outcome):
        if self.coverage:
            self.coverage.verified(action,before,after)
            if action.kind==Kind.BUY and any(c.entity_id==action.source and c.card_type in ('SPELL','BATTLEGROUND_SPELL') for c in before.shop):
                self.coverage.done[(before.game_id,'all','buy_spell',None)]+=1

    def checkpoint_state(self):
        if not self.coverage:return None
        from dataclasses import asdict
        return dict(profile=asdict(self.coverage.profile),done=[[*k,v] for k,v in self.coverage.done.items()],frozen_turn=self.coverage.frozen_turn)

    def restore_checkpoint_state(self,state):
        if not self.coverage:raise ValueError('coverage_checkpoint_requires_coverage_policy')
        from dataclasses import asdict
        if state['profile']!=asdict(self.coverage.profile):raise ValueError('checkpoint_policy_profile_mismatch')
        self.coverage.done=Counter({tuple(row[:-1]):row[-1] for row in state['done']})
        self.coverage.frozen_turn=state['frozen_turn']
    @staticmethod
    def score(card):
        return (card.attack or 0) + (card.health or 0) + 0.1 * (card.tier or 0)

    @staticmethod
    def first(obs,kind):
        actions=[a for a in obs.actions if a.kind==kind]
        if kind in (Kind.PLAY,Kind.ACTIVATE,Kind.HERO_POWER):
            board={c.entity_id:c for c in obs.board}
            friendly=[a for a in actions if a.target in board]
            if friendly:return max(friendly,key=lambda a:RulePolicy.score(board[a.target]))
        return actions[0] if actions else None

    def decide(self, obs: Observation) -> Decision:
        if self.coverage:obs=self.coverage.filter(obs)
        def decision(action, why):
            return Decision(action, obs.revision, why)

        def first(kind):
            return self.first(obs,kind)

        if obs.mode != "GT_BATTLEGROUNDS" and obs.source != "sandbox":
            return decision(Action(Kind.WAIT), "仅支持单人酒馆，当前模式未确认或不受支持")
        if obs.phase == Phase.COMBAT:
            return decision(Action(Kind.WAIT), "战斗阶段等待")
        if obs.phase == Phase.GAME_OVER:
            return decision(first(Kind.DISMISS) or Action(Kind.WAIT), "对局结束")
        for kind in (Kind.TARGET,Kind.CONFIRM,Kind.CANCEL):
            if a:=first(kind):return decision(a,"处理当前适配器已确认的强制交互")
        if obs.phase == Phase.CHOICE:
            choose = [a for a in obs.actions if a.kind in (Kind.CHOOSE,Kind.SELECT)]
            if choose:
                scores = {c.entity_id: self.score(c) for c in obs.choices}
                return decision(max(choose, key=lambda a: scores.get(a.source, 0)), "选择当前评分最高的合法选项")
            return decision(Action(Kind.WAIT), "此选择需要专用适配，暂停而非猜测")
        if self.coverage and (selected:=self.coverage.priority(obs,self.score)):return selected
        for kind in (Kind.ENTER, Kind.PLAY):
            if a := first(kind):
                return decision(a, "优先处理入口或可打出的手牌")
        # Deliberately simple curve, never invent an upgrade action or its cost.
        for kind in (Kind.HERO_POWER,Kind.ACTIVATE):
            if a:=first(kind):return decision(a,'使用当前可执行且服务器允许的主动能力')
        if (a := first(Kind.UPGRADE)) and (obs.gold or 0) >= 4 and len(obs.board) >= 2:
            return decision(a, "有基础场面且服务器允许，升级酒馆")
        cards = {c.entity_id: c for c in obs.shop}
        if len(obs.board) >= 7 and obs.hand:
            sells = {a.source: a for a in obs.actions if a.kind == Kind.SELL}
            weakest = min(obs.board, key=self.score)
            minions = [c for c in obs.hand if c.card_type == "MINION"]
            if minions and weakest.entity_id in sells and (len(obs.hand) >= 9 or max(map(self.score,minions)) > self.score(weakest)):
                return decision(sells[weakest.entity_id], "满场时为手中随从腾出位置")
        buys = [a for a in obs.actions if a.kind == Kind.BUY and a.source in cards and cards[a.source].card_type == "MINION"]
        if buys and len(obs.board) < 7:
            return decision(max(buys, key=lambda a: self.score(cards[a.source])), "购买合法选项中身材评分最高的随从")
        if len(obs.board) >= 7 and buys and (obs.gold or 0) >= 3:
            sells = {a.source: a for a in obs.actions if a.kind == Kind.SELL}
            weakest = min(obs.board, key=self.score)
            if weakest.entity_id in sells and max(self.score(cards[a.source]) for a in buys) > self.score(weakest) + 2:
                return decision(sells[weakest.entity_id], "腾出最弱随从的位置以改善场面")
        desired=tuple(c.entity_id for c in sorted(obs.board,key=lambda c:(-self.score(c),c.entity_id)))
        current=tuple(c.entity_id for c in obs.board)
        if current != desired:
            moves=[a for a in obs.actions if a.kind==Kind.REORDER and a.scope in (None,"board")]
            exact=next((a for a in moves if a.order==desired),None)
            if exact:return decision(exact,"将场上随从整理为固定身材顺序")
            mismatch=next(i for i,(a,b) in enumerate(zip(current,desired)) if a!=b)
            move=next((a for a in moves if a.source==desired[mismatch] and a.position==mismatch+1),None)
            if move:return decision(move,"逐步整理场上顺序，已排好后不重复换位")
        spell_buys=[a for a in obs.actions if a.kind==Kind.BUY and a.source in cards and cards[a.source].card_type in ("SPELL","BATTLEGROUND_SPELL")]
        if spell_buys:return decision(spell_buys[0],"购买当前允许的酒馆法术，后续按合法目标打出")
        if a:=first(Kind.FREEZE):
            frozen=bool(obs.shop) and all(c.tags.get('FROZEN')=='1' for c in obs.shop)
            weakest=min(map(self.score,obs.board),default=0)
            keep=bool(obs.shop and (obs.gold or 0)<3 and any(c.card_type=='MINION' and self.score(c)>weakest+2 for c in obs.shop))
            if keep!=frozen:return decision(a,'保留下一回合的明显场面提升' if keep else '取消不再需要的锁店')
        for kind,why in ((Kind.REFRESH,"没有合适购买，刷新商店"),(Kind.END,"结束当前受控回合")):
            if a:=first(kind):return decision(a,why)
        return decision(Action(Kind.WAIT), "没有已确认且适合当前规则策略的动作")
