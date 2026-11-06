"""Portable, deterministic action-ranking agent. No client, input or clock APIs.

Utilities are estimates, not combat probabilities or learned win rates. The
candidate set is supplied by the environment's legal resolver. Every decision
is recomputed after the environment verifies the preceding action.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import math
import re

from .model import Action, Decision, Kind, Phase
from .strategy_catalog import bundled_catalog, TRIBES


@dataclass(frozen=True)
class HeuristicConfig:
    version: str = 'heuristic-1.3'
    ability_uses_per_source_turn: int = 1
    gift_uses_per_turn: int = 1
    max_refreshes_per_turn: int = 12
    max_economic_actions_per_turn: int = 60

    def __post_init__(self):
        if any(v<=0 for k,v in asdict(self).items() if k!='version'):
            raise ValueError('heuristic_budgets_must_be_positive')


@dataclass(frozen=True)
class RankedAction:
    action: Action
    utility: float
    reason: str


class HeuristicPolicy:
    def __init__(self, config=None, catalog=None):
        self.config=config or HeuristicConfig()
        self.catalog=catalog or bundled_catalog()
        self.counts=Counter()
        self.game_id=None

    def metadata(self):
        return dict(policy=asdict(self.config),catalog_sha256=self.catalog.digest,
                    catalog_builds=self.catalog.builds,catalog_source=self.catalog.source)

    def checkpoint_state(self):
        return dict(schema=1,metadata=self.metadata(),game_id=self.game_id,
                    counts=[[*key,value] for key,value in sorted(self.counts.items(),key=lambda row:str(row[0]))])

    def restore_checkpoint_state(self, state):
        # JSON converts tuples to lists: compare normalized payloads.
        import json
        if state.get('schema')!=1 or json.dumps(state.get('metadata'),sort_keys=True)!=json.dumps(self.metadata(),sort_keys=True):
            raise ValueError('heuristic_checkpoint_policy_or_catalog_mismatch')
        self.game_id=state['game_id']
        self.counts=Counter({tuple(row[:-1]):row[-1] for row in state['counts']})

    def count(self, obs, kind, source=None):
        return self.counts[(obs.game_id,obs.turn,str(kind),source)]

    def on_verified(self, action, before, after, outcome):
        # Called by AgentController only after its strict effect verifier passes.
        if str(outcome.status) not in ('confirmed','resolved'):return
        if self.game_id!=before.game_id:self.counts.clear();self.game_id=before.game_id
        for source in {None,action.source}:
            self.counts[(before.game_id,before.turn,str(action.kind),source)]+=1
        if action.kind==Kind.REORDER and 'placement_deviated' in outcome.proofs:
            self.counts[(before.game_id,before.turn,'reorder_deviation',None)]+=1
        if action.kind==Kind.ACTIVATE and any(c.entity_id==action.source and c.card_id=='BG36_Button_DarkGift' for c in before.controls):
            self.counts[(before.game_id,before.turn,'dark_gift',None)]+=1
            self.counts[(before.game_id,'all','dark_gift',None)]+=1
        if action.kind==Kind.FREEZE:
            frozen=bool(before.shop) and all(c.tags.get('FROZEN')=='1' for c in before.shop)
            self.counts[(before.game_id,before.turn,'unlock' if frozen else 'lock',None)]+=1
        self.counts[(before.game_id,before.turn,'economic',None)]+=int(action.kind in
            (Kind.BUY,Kind.SELL,Kind.PLAY,Kind.REFRESH,Kind.ACTIVATE,Kind.HERO_POWER))
        # Bounded memory without modifying repeated decide() calls.
        for key in list(self.counts):
            if key[0]!=before.game_id or (isinstance(key[1],int) and key[1]<(before.turn or 0)-2):del self.counts[key]

    def text(self,c,o):return self.catalog.text(c,o.build) if c else ''
    def tribes(self,c,o):return self.catalog.tribes(c,o.build)
    def has(self,c,o,tag):return self.catalog.mechanic(c,o.build,tag)

    def body(self,c,o):
        meta=self.catalog.get(c,o.build)
        a=max(0,c.attack if c.attack is not None else meta.get('attack',0))
        h=max(0,c.health if c.health is not None else meta.get('health',0))
        value=math.sqrt(a*h)*1.4+.35*(a+h)
        if self.has(c,o,'DIVINE_SHIELD'):value+=max(3,a*.9)
        if self.has(c,o,'VENOMOUS') or self.has(c,o,'POISONOUS'):value+=max(7,math.sqrt(h)*3)
        if self.has(c,o,'WINDFURY'):value+=a*.45
        if self.has(c,o,'REBORN'):value+=2+a*.5
        text=self.text(c,o)
        if 'adjacent minion' in text and ('damage' in text or 'attacks' in text):value+=a*.7
        if self.has(c,o,'DEATHRATTLE'):
            value+=3
            if 'summon' in text:value+=3
            if 'destroy the minion that killed' in text:value+=12
        return value

    def horizon(self,o):
        hp=o.hero.health if o.hero and o.hero.health is not None else 30
        return max(1,min(4,(hp+int(o.hero.tags.get('ARMOR',0)) if o.hero else hp)/10))

    def engines(self,c,o):
        """Recognize reusable trigger families, not card-specific input workflows."""
        text=self.text(c,o)
        if c.tags.get('SILENCED')=='1':return 0
        board=tuple(x for x in o.board if x.entity_id!=c.entity_id)
        friendly=sum(bool(self.tribes(c,o)&self.tribes(x,o)) for x in board)
        triggers=('after you','whenever you','at the end of your turn','start of combat','avenge')
        repeating=any(t in text for t in triggers)
        stats=any(t in text for t in ('give','gain','improves','extra','consume'))
        affected=max(1,friendly) if any(t.lower() in text for t in TRIBES) else max(1,len(board))
        # Trigger enablers are valued only with the relevant actual board.
        if 'deathrattles trigger' in text:return 5*sum(self.has(x,o,'DEATHRATTLE') for x in board)
        if 'end of turn effects trigger' in text:return 5*sum('end of your turn' in self.text(x,o) for x in board)
        if 'battlecries trigger' in text:return 4*sum(self.has(x,o,'BATTLECRY') for x in board+o.hand)
        value=0
        if repeating and stats:value+=self.horizon(o)*min(affected,5)*1.6
        income=self.recurring_value(c,o)
        if repeating and not income and any(t in text for t in ('get a','get 2','gold','reduce','cost')):value+=self.horizon(o)*3
        value+=income
        if 'spellcraft' in text:value+=3*self.horizon(o)
        if 'battlecry' in text and any(t in text for t in ('get','gold','give','discover')):value+=4
        # Off-tribe single engines remain possible pivot candidates, but do not
        # receive the same reward as a supported composition.
        return value

    def recurring_value(self,c,o):
        """Small discounted-income proxy, including known log attachments.

        This is deliberately bounded; it does not simulate future shop rolls.
        Unobserved gift effects are not invented from the minion's identity.
        """
        texts=[self.text(c,o)]+[self.text(e,o) for e in o.controls
            if str(e.tags.get('ATTACHED',''))==str(c.entity_id)]
        value=0;h=self.horizon(o)
        for text in texts:
            period_match=re.search(r'every (\d+) turns',text)
            periodic=period_match or 'end of your turn' in text or 'start of your turn' in text
            period=int(period_match.group(1)) if period_match else 1
            if periodic and ('get' in text or 'add' in text) and ('minion' in text or 'copy of this' in text):
                value+=h/period*3*2.1*.85
            refresh=re.search(r'(\d+) free refresh',text)
            if refresh and ('deathrattle' in text or periodic):
                value+=int(refresh.group(1))*h*.65*1.8
        return min(24,value)

    def board_security(self,o):
        # Own-board risk proxy, not an opponent win-probability estimate.
        turn=o.turn or 1
        reference=6.5*min(7,max(1,turn-1))*(1+.15*max(0,turn-3))
        return min(2,sum(self.body(c,o) for c in o.board)/reference)

    def value(self,c,o):return self.body(c,o)+self.engines(c,o)

    def copies(self,c,o):
        if not c.card_id or self.catalog.golden(c,o.build):return 0
        return sum(x.card_id==c.card_id and not self.catalog.golden(x,o.build) for x in o.board+o.hand)

    def support_bonus(self,c,o):
        # Prefer meaningful supported synergies, never force a tribe on turn 1.
        tribes=self.tribes(c,o)
        supported=sum(bool(tribes&self.tribes(x,o)) for x in o.board)
        base=max(0,supported-1)*1.2
        # Cross-card trigger compatibility makes a composition emerge from
        # actual enablers instead of collecting seven cards of one tribe.
        for engine in o.board:
            text=self.text(engine,o)
            if 'after you play' in text:
                if any(t.lower() in text and t in tribes for t in TRIBES):base+=4*self.horizon(o)
                if 'battlecry' in text and self.has(c,o,'BATTLECRY'):base+=4
            if 'deathrattles trigger' in text and self.has(c,o,'DEATHRATTLE'):base+=5
            if 'end of turn effects trigger' in text and 'end of your turn' in self.text(c,o):base+=6
            if 'magnetiz' in text and self.has(c,o,'MODULAR'):base+=6
        return base

    def replacement_value(self,c,o):
        immediate=0
        if self.has(c,o,'BATTLECRY'):
            # Estimate only the battlecry tail, including board-wide buffs;
            # do not interpret a persistent engine's whole text as a spell.
            immediate=max(0,self.effect(c,o,battlecry_only=True))*.6
        return self.value(c,o)+self.support_bonus(c,o)+immediate

    def acquisition(self,c,o):
        weakest=min((self.value(x,o) for x in o.board),default=0)
        base=self.replacement_value(c,o)-(weakest if len(o.board)>=7 else 0)
        copies=self.copies(c,o)
        base+=18 if copies>=2 else 2 if copies==1 else 0
        base+=12 if len(o.board)<4 else 5 if len(o.board)<7 else 0
        return base

    def effect(self,c,o,target=None,*,battlecry_only=False):
        """Conservative text-feature estimate. No rule execution or legality here."""
        text=self.text(c,o)
        if battlecry_only:
            parts=re.split(r'battlecry(?: and deathrattle)?:',text,maxsplit=1)
            text=parts[-1] if len(parts)>1 else ''
        if not text:return -4  # Unknown spells/abilities are not blindly cast.
        value=0
        if 'discover' in text:value+=10+2*(o.tavern_tier or 1)
        if any(t in text for t in ('get a','get 2','gold coin','gain 1 gold','gain 2 gold')):value+=7
        # 附魔链索等明确的“从酒馆随机偷一个随从”是获得随从收益，不能因
        # 文本没有 get/add 被当作零收益并在搜索前过滤。随机结果取公开均值，
        # 不假定可以任选最强，也不扩展到偷对手/未知对象等不同机制。
        if (not battlecry_only and c.card_type in ('SPELL','BATTLEGROUND_SPELL','HERO_POWER')
                and re.search(r'\bsteal a random minion from (?:the|your) tavern\b',' '.join(text.split()))):
            offered=[x for x in o.shop if x.card_type=='MINION']
            if offered:value+=sum(self.acquisition(x,o) for x in offered)/len(offered)
        buffs=re.findall(r'\+(\d+)\s*/\s*\+(\d+)',text)
        if buffs:
            buff=sum(int(a)+int(h) for a,h in buffs)
            n=len(o.board) if ('your minions' in text or 'all friendly' in text) else 1
            if 'minions of' in text or 'share a type' in text:
                n=sum(bool(self.tribes(target,o)&self.tribes(x,o)) for x in o.board) if target else 1
            if not target and not o.board and 'tavern' not in text:n=0
            value+=buff*n*.6
        attack_buff=re.search(r'(?:have|give[^.]*?)\s*\+(\d+) attack',text)
        if attack_buff:value+=int(attack_buff.group(1))*sum('undead' not in text or 'UNDEAD' in self.tribes(x,o) for x in o.board)*.6
        health_buff=re.search(r'(?:give|gain|have)[^.]*?\+(\d+) health',text)
        if health_buff and 'minion' in text:
            n=len(o.board) if 'your minions' in text or 'all friendly' in text else int(bool(target or o.board))
            value+=int(health_buff.group(1))*n*.6
        if 'divine shield' in text:
            value+=max(3,(target.attack or 0)*.8) if target and not self.has(target,o,'DIVINE_SHIELD') else 0
        if 'golden' in text:value+=12+(self.engines(target,o) if target else 0)
        # A future/progress reward is not awarded on every click. For example,
        # Dig's eventual Golden minion must not crowd out a turn-one purchase.
        progress=re.search(r'(\d+) digs? left',text)
        if progress:value/=max(1,int(progress.group(1)))
        if 'refresh' in text:value+=3
        if 'copy' in text and ('get' in text or 'add' in text):value+=max(5,self.value(target,o)*.6) if target else 7
        if 'consume' in text and target:
            value+=sum(self.body(x,o) for x in sorted(o.shop,key=lambda x:self.body(x,o),reverse=True)[:2])*.6
        if 'destroy' in text:
            if target and target in o.board:value-=self.value(target,o)
            elif target is None:value-=12
        if 'transform' in text and target:value+=max(0,12-self.engines(target,o))
        if ('give' in text or 'gain' in text) and target and target in o.shop and 'tavern' not in text and 'consume' not in text:
            value-=12
        if 'damage' in text and ('your hero' in text or 'take ' in text):
            hp=o.hero.health if o.hero else 30
            value-=12 if hp is not None and hp<12 else 3
        # Spells also trigger existing scaling engines (Azerite, Czarina, etc.).
        value+=sum(2 for x in o.board if 'cast' in self.text(x,o) and ('spell' in self.text(x,o)))
        if target:
            if target in o.board and buffs:
                value+=int(self.has(target,o,'DIVINE_SHIELD'))*3+int(self.has(target,o,'WINDFURY'))*2
                # Separate attack/health branches instead of tying on total
                # stats; shield carriers leverage attack, Venom carriers health.
                value+=sum(int(a)*.25 for a,h in buffs) if self.has(target,o,'DIVINE_SHIELD') else 0
                value+=sum(int(h)*.2 for a,h in buffs) if self.has(target,o,'VENOMOUS') else 0
            if 'destroy' in text and target in o.board and self.has(target,o,'DEATHRATTLE'):value+=3
        return value

    @staticmethod
    def price(a,c=None):
        # Server purchase cost is authoritative; never treat minion mana COST
        # as a gold price. Unknown purchase costs are deliberately not budgeted.
        if a.resource_cost is not None:return a.resource_cost
        if a.kind==Kind.BUY:return None
        if a.kind==Kind.ACTIVATE and c and 'INTERACTABLE_OBJECT_COST' in c.tags:
            try:return int(c.tags['INTERACTABLE_OBJECT_COST'])
            except (ValueError,TypeError):return None
        return c.cost if c and c.cost is not None else 0

    @staticmethod
    def minion_price(o):
        prices=[a.resource_cost for a in o.actions if a.kind==Kind.BUY and a.resource_cost is not None
                and any(c.entity_id==a.source and c.card_type=='MINION' for c in o.shop)]
        prices += [c.cost for c in o.controls if 'dragbuy' in c.card_id.lower() and c.cost is not None]
        return min(prices) if prices else None

    def action_effect(self,a,c,o,target=None):
        if a.sub_option_id is None:return self.effect(c,o,target)
        selected=next((s.target for s in a.interaction.stages if s.mode=='branch' and s.operation=='choose'),None) if a.interaction else None
        branch=next((b for b in o.branches if b.entity_id==selected),None)
        if branch and self.text(branch,o):return self.effect(branch,o,target)
        # Older observations lack branch entities. Do not add mutually exclusive
        # parent alternatives together as though the action granted both.
        count=sum(s.mode=='branch' and s.operation=='candidate' for s in a.interaction.stages) if a.interaction else 1
        return self.effect(c,o,target)/max(1,count)

    def preservation(self,c,o):
        """Value of keeping this offer, excluding the generic empty-slot bonus."""
        floor=min((self.value(x,o) for x in o.board),default=0) if len(o.board)>=7 else 0
        triple=18 if self.copies(c,o)>=2 else 0
        return self.replacement_value(c,o)-floor+triple

    def desired_order(self,o):
        def priority(c):
            text=self.text(c,o)
            front=self.body(c,o)
            if 'summon' in text and self.has(c,o,'DEATHRATTLE'):front+=15
            if 'destroy the minion that killed' in text:front+=25
            if self.has(c,o,'VENOMOUS'):front+=10
            if 'adjacent minion' in text and 'damage' in text:front+=20
            if 'avenge' in text or 'deathrattles trigger' in text:front-=40
            if self.has(c,o,'TAUNT'):front-=5
            return (-front,c.entity_id)
        order=list(sorted(o.board,key=priority))
        # Young Murk-Eye family: battlecry neighbors, including end positions.
        for engine in list(order):
            if 'battlecries of adjacent' not in self.text(engine,o):continue
            candidates=sorted((x for x in order if x!=engine and self.has(x,o,'BATTLECRY')),key=lambda c:(-self.engines(c,o),c.entity_id))[:2]
            if candidates:
                order=[x for x in order if x!=engine and x not in candidates]
                order=candidates[:1]+[engine]+candidates[1:]+order
        return tuple(c.entity_id for c in order)

    def rank(self,o):
        if not o.ready or (o.mode!='GT_BATTLEGROUNDS' and o.source!='sandbox') or o.phase in (Phase.COMBAT,Phase.UNKNOWN):return ()
        cards={c.entity_id:c for c in o.board+o.shop+o.hand+o.choices+o.controls+((o.hero,) if o.hero else ())}
        gold=o.gold or 0;ranked=[]
        weakest=min(o.board,key=lambda c:(self.value(c,o),c.entity_id),default=None)
        desired=self.desired_order(o)
        current=tuple(c.entity_id for c in o.board)
        budget=self.count(o,'economic')>=self.config.max_economic_actions_per_turn
        for a in o.actions:
            c=cards.get(a.source);target=cards.get(a.target);cost=self.price(a,c)
            score=-10000.;why='不适合当前状态'
            if a.kind==Kind.ENTER and o.phase==Phase.MENU:score=1000;why='进入单局'
            elif a.kind==Kind.DISMISS and o.phase==Phase.GAME_OVER:score=1000;why='完成结算返回菜单'
            elif a.kind in (Kind.CHOOSE,Kind.SELECT) and o.phase==Phase.CHOICE:
                if a.source in (o.interaction.selected if o.interaction else ()):continue
                score=1000+(self.value(c,o) if c and c.card_type=='MINION' else self.effect(c,o))
                if c and c.card_type=='HERO':
                    hp=self.catalog.get(c,o.build).get('heroPowerDbfId')
                    power_id=self.catalog.by_db.get(hp)
                    from .model import Card
                    score=1000+float(c.tags.get('ARMOR',self.catalog.get(c,o.build).get('armor',0)))+self.effect(Card(-1,power_id or ''),o)
                if c and c.tags.get('QUEST')=='1':
                    from .interaction_protocol import quest_reward_links
                    # Semantic link only; no coordinates enter the policy.
                    reward=quest_reward_links(o).get(c.entity_id)
                    if reward:score=1000+self.effect(reward,o)
                if a.resource_cost is not None:score-=a.resource_cost*2
                why='强制选择：战力、成长、经济与当前阵容适配'
            elif a.kind==Kind.TARGET:
                score=1200+self.effect(c,o,target)
                if target and not self.text(c,o):score+=self.body(target,o)*.01
                why='完成已授权的目标交互'
            elif a.kind==Kind.CONFIRM:
                more=bool(o.interaction and len(o.interaction.selected)<o.interaction.max_select and
                    any(x.kind==Kind.SELECT and x.source not in o.interaction.selected for x in o.actions))
                score=900 if more else 1100;why='提交合法的已完成选择'
            elif a.kind==Kind.CANCEL:score=1;why='无更好合法分支时退出可取消交互'
            elif o.phase!=Phase.RECRUIT:continue
            elif a.kind==Kind.PLAY and c and not budget:
                if c.card_type=='MINION':
                    if len(o.board)<7 or (a.interaction and a.interaction.placement=='magnetize'):
                        score=40+self.value(c,o)
                        if target and self.has(c,o,'BATTLECRY'):score+=self.effect(c,o,target,battlecry_only=True)
                        if target and a.interaction and a.interaction.placement=='magnetize':score+=self.body(target,o)*.01
                else:
                    effect=self.action_effect(a,c,o,target)
                    score=20+effect if effect>0 else -10000
                why='兑现手牌战力或有正收益的法术'
            elif a.kind==Kind.BUY and c and cost is not None and not budget:
                score=(self.acquisition(c,o) if c.card_type=='MINION' else self.effect(c,o,target))-cost*2
                if c.card_type=='MINION' and len(o.hand)>=9 and self.copies(c,o)<2:score-=30
                if c.card_type!='MINION' and len(o.hand)>=8:score-=20
                # Do not accumulate weak minions that cannot replace the board.
                if c.card_type=='MINION' and len(o.board)>=7 and self.acquisition(c,o)<3:score=-10000
                why='购买：净提升、经济触发、三连与金币机会成本'
            elif a.kind==Kind.SELL and c and weakest and not budget:
                best_hand=max((self.replacement_value(x,o) for x in o.hand if x.card_type=='MINION'),default=-10000)
                if len(o.board)>=7 and best_hand>self.value(c,o)+2:
                    score=35+best_hand-self.value(c,o);why='为更强的手牌腾出场面'
                elif c.entity_id==weakest.entity_id:
                    purchase_price=self.minion_price(o)
                    affordable=[x for x in o.shop if x.card_type=='MINION' and purchase_price is not None and gold<purchase_price<=gold+1]
                    best=max((self.acquisition(x,o)-self.value(c,o) for x in affordable),default=-10000)
                    if best>8 and len(o.board)>2:score=best-4;why='一金币缺口：出售弱随从换明确提升'
            elif a.kind==Kind.UPGRADE and not budget:
                price=a.resource_cost if a.resource_cost is not None else o.upgrade_cost
                if price is not None:
                    tier=o.tavern_tier or 1;turn=o.turn or 1
                    target_tier=min(6,1+(turn+1)//2)
                    score=8+max(0,target_tier-tier)*7-price*1.8
                    if len(o.board)<min(5,turn):score-=12
                    if self.horizon(o)<2:score-=15
                    if tier>=5:score-=5
                    if tier==1 and turn>=2 and o.board:score+=18
                    if turn>=3 and len(o.board)<3:score-=8*(3-len(o.board))
                    security=self.board_security(o)
                    score+=max(0,security-1)*12*min(1,self.horizon(o)/3)
                    score-=max(0,1-security)*16
                    why='升本：成长窗口与当前战力、血量和花费的权衡'
            elif a.kind in (Kind.ACTIVATE,Kind.HERO_POWER) and c and not budget:
                limit=self.config.ability_uses_per_source_turn
                if c.card_id=='BG36_Button_DarkGift':
                    limit=self.config.gift_uses_per_turn
                    # Save some opportunities for higher-tier offerings, but
                    # convert gifts early when short on tempo or low on health.
                    score=15+max(0,(o.turn or 0)-5)*1.5+(8 if len(o.board)<5 else 0)-2*(cost or 0)+self.horizon(o)*2
                    if (o.turn or 0)<5 and len(o.board)>=3 and self.horizon(o)>2:score-=14
                    if len(o.hand)>=9 or self.counts[(o.game_id,'all','dark_gift',None)]>=3:score=-10000
                    why='黑暗之赐：即时补强与后期发现质量的权衡'
                else:
                    effect=self.action_effect(a,c,o,target)
                    score=3+effect-2*(cost or 0) if effect>0 else -10000
                    if effect>0 and cost==0:score+=6
                    why='使用具有可估计收益的英雄技能或发动能力'
                uses=self.count(o,'dark_gift') if c.card_id=='BG36_Button_DarkGift' else self.count(o,a.kind,a.source)
                if uses>=limit:score=-10000
            elif a.kind==Kind.REORDER and a.scope in (None,'board') and desired!=current and not self.count(o,'reorder_deviation'):
                mismatch=next(i for i,(x,y) in enumerate(zip(current,desired)) if x!=y)
                if a.order==desired or (a.source==desired[mismatch] and a.position==mismatch+1):
                    score=3;why='安排攻击顺序并保护成长引擎'
            elif a.kind==Kind.FREEZE:
                frozen=bool(o.shop) and all(c.tags.get('FROZEN')=='1' for c in o.shop)
                good=[c for c in o.shop if c.card_type=='MINION' and self.preservation(c,o)>=16]
                purchase_price=self.minion_price(o)
                keep=bool(good and purchase_price is not None and gold<purchase_price and len(o.hand)<9)
                if keep!=frozen and self.count(o,'unlock' if frozen else 'lock')<1:score=2;why='仅保留核心或三连等高价值目标；普通补场不锁店'
            elif a.kind==Kind.REFRESH and not budget and self.count(o,Kind.REFRESH)<self.config.max_refreshes_per_turn:
                # Keep frozen opportunities and never roll while a minion waits
                # for a mapper. An uncastable spell does not block the turn.
                if not any(c.card_type=='MINION' for c in o.hand) and gold>=1 and not (o.shop and all(c.tags.get('FROZEN')=='1' for c in o.shop)):
                    score=.5;why='无正收益购买或升本，搜索更有价值的选项'
            elif a.kind==Kind.END:score=0;why='本回合没有更高收益的已确认动作'
            if math.isfinite(score) and score>-10000:ranked.append(RankedAction(a,score,why))
        # Do not strand the gold needed for a productive purchase just to use
        # a paid ability. Evaluate the actual legal price, not nominal COST.
        # Leftover gold does not carry into the next recruit phase. Discount
        # its opportunity cost only for a known beneficial spell when there is
        # no competing productive purchase, upgrade or paid ability.
        residual=[]
        for r in ranked:
            a=r.action;c=cards.get(a.source);cost=self.price(a,c)
            if a.kind==Kind.BUY and c and c.card_type!='MINION' and cost and cost==gold and self.effect(c,o)>0:
                competition=False
                for other in ranked:
                    if other is r or other.utility<=0 or other.action.kind not in (Kind.BUY,Kind.UPGRADE,Kind.ACTIVATE,Kind.HERO_POWER):continue
                    price=other.action.resource_cost if other.action.resource_cost is not None else o.upgrade_cost if other.action.kind==Kind.UPGRADE else self.price(other.action,cards.get(other.action.source))
                    if price is not None and 0<price<=gold:competition=True;break
                if not competition:r=RankedAction(a,r.utility+cost*1.75,r.reason+'；剩余金币无更高收益用途，降低机会成本')
            residual.append(r)
        ranked=residual
        buys=[r for r in ranked if r.action.kind==Kind.BUY and r.utility>0]
        adjusted=[]
        for r in ranked:
            a=r.action;c=cards.get(a.source);cost=self.price(a,c)
            if a.kind in (Kind.HERO_POWER,Kind.ACTIVATE) and cost and len(o.board)<5 and c and c.card_id!='BG36_Button_DarkGift':
                blocked=[b.utility for b in buys if b.action.resource_cost is not None and gold-cost<b.action.resource_cost<=gold]
                if blocked:r=RankedAction(a,r.utility-max(blocked),r.reason+'；扣除阻断购买的机会成本')
            adjusted.append(r)
        return tuple(sorted(adjusted,key=lambda r:(-r.utility,r.action.key)))

    def decide(self,o):
        ranked=self.rank(o)
        if not ranked or ranked[0].utility<0:
            return Decision(Action(Kind.WAIT),o.revision,'等待状态推进；没有正收益的已确认合法动作')
        best=ranked[0]
        return Decision(best.action,o.revision,f'{self.config.version}: {best.reason}; utility={best.utility:.2f}; catalog={"matched" if self.catalog.compatible(o.build) else "live-only"}')

    def preview(self,o):
        # rank/decide only read counters. on_verified remains controller-only.
        return self.decide(o)
