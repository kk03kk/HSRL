"""Incremental Power.log projection, inspired by the documented hslog protocol.

Not a rules engine. Retains unknown tags. Server option lists are authoritative
evidence; no hard-coded gold curve, card costs or fabricated legal targets.
"""
from __future__ import annotations

from collections import Counter,deque
from dataclasses import dataclass, field
import re
import time

from .model import Action, Card, Kind, Observation, Phase,Resolution, InteractionContext,InteractionSpec,InteractionStage,FrozenTags

LINE = re.compile(r"^[DWE] [\d:.]+ (?P<method>[\w.]+)\(\) -\s*(?P<body>.*)$")
REF = re.compile(r"\[entityName=(.*?) id=(\d+) zone=(\w+) zonePos=(\d+) cardId=(\w*) player=(\d+)\]")
TAG = re.compile(r"tag=(\w+) value=([^\s]+)")
NUM_TYPES = {3: "HERO", 4: "MINION", 5: "SPELL", 10: "HERO_POWER", 2: "PLAYER", 1: "GAME",42:"BATTLEGROUND_SPELL",44:"BATTLEGROUND_TRINKET",12:"GAME_MODE_BUTTON"}
HAND_TYPES = ("MINION","SPELL","BATTLEGROUND_SPELL")
NUM_ZONES = {1: "PLAY", 2: "DECK", 3: "HAND", 4: "GRAVEYARD", 5: "REMOVEDFROMGAME", 6: "SETASIDE"}


def integer(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class VersionedTags(dict):
    """Dirty tracking makes unchanged entity views reusable and immutable."""
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.version=0
    def __setitem__(self,key,value):
        if key not in self or self[key]!=value:self.version+=1
        super().__setitem__(key,value)
    def __delitem__(self,key):super().__delitem__(key);self.version+=1
    def update(self,*a,**kw):
        for key,value in dict(*a,**kw).items():self[key]=value
    def setdefault(self,key,value=None):
        if key not in self:self[key]=value
        return self[key]
    def clear(self):
        if self:self.version+=1
        super().clear()
    def pop(self,key,*default):
        if key in self:self.version+=1
        return super().pop(key,*default)
    def popitem(self):
        result=super().popitem();self.version+=1;return result
    def __ior__(self,other):self.update(other);return self


@dataclass
class Entity:
    id: int
    card_id: str = ""
    name: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    _cached_key: tuple|None=field(default=None,init=False,repr=False,compare=False)
    _cached_card: Card|None=field(default=None,init=False,repr=False,compare=False)

    def __post_init__(self):self.tags=VersionedTags(self.tags)

    def card(self) -> Card:
        if not isinstance(self.tags,VersionedTags):self.tags=VersionedTags(self.tags);self._cached_key=None
        key=(self.card_id,self.name,self.tags.version)
        if key==self._cached_key:return self._cached_card
        health = integer(self.tags.get("HEALTH"))
        if health is not None:
            health -= integer(self.tags.get("DAMAGE"), 0)
        ct = self.tags.get("CARDTYPE", "")
        zone = self.tags.get("ZONE", "")
        result=Card(self.id, self.card_id, self.name,
                    integer(self.tags.get("CONTROLLER")), NUM_ZONES.get(integer(zone), zone),
                    integer(self.tags.get("ZONE_POSITION"), 0), NUM_TYPES.get(integer(ct), ct),
                    integer(self.tags.get("ATK")), health,
                    integer(self.tags.get("TECH_LEVEL")), integer(self.tags.get("COST")), FrozenTags(self.tags))
        self._cached_key=key;self._cached_card=result
        return result


class PowerState:
    def __init__(self, player_id: int | None = None):
        self.forced_player = player_id
        self.local_player_id = player_id
        self.entities: dict[int, Entity] = {}
        self.players: dict[int, int] = {}
        self.aliases: dict[str, int] = {}
        self.pending_alias_tags: dict[str, dict] = {}
        self.game_entity: int | None = None
        self.mode = "unknown"
        self.build = None
        self.scene = None
        self.revision = 0
        self.generation = 0
        self.active_entity: int | None = None
        self.options: list[dict] = []
        self.options_pending=False
        self.current_option: dict | None = None
        self.current_sub_option: dict | None = None
        self.choice: dict | None = None
        self.choice_rows: dict[int, int] = {}
        self.power_provider: str | None = None
        self.stats = Counter()
        self.last_error: str | None = None
        self.block_stack=[]
        self.resolutions=deque(maxlen=128)
        self.chosen_context=None
        self.power_lines=0;self.chosen_log_position=None
        self.chosen_rows={};self.chosen_count=None;self.chosen_owner=None
        self.projection_invalid_reason=None
        self.choice_display={}
        from .client_objects import ClientObjects
        self.client_objects=ClientObjects()
        self.client_task_list=None
        self.client_submission_position=None
        self.client_playback_position=None

    def _finish_block(self,entry,complete):
        frame=entry["action"]
        if frame:
            if complete:
                from dataclasses import replace
                observation=replace(self.observe(),resolutions=(),actions=())
                keep={'FROZEN','EXHAUSTED','INTERACTABLE_OBJECT_COST','PREMIUM','MAGNETIC','CARDRACE','ATTACHED','CREATOR',
                      'TAG_SCRIPT_DATA_NUM_1','TAG_SCRIPT_DATA_NUM_2'}
                from .verification import ABILITY_PROGRESS_TAGS
                keep.update(ABILITY_PROGRESS_TAGS)
                fields={zone:tuple(replace(c,tags={k:v for k,v in c.tags.items() if k in keep}) for c in getattr(observation,zone))
                    for zone in ('hand','board','shop','choices','controls')}
                view=replace(observation,**fields).to_dict()
                # Never embed older receipts recursively in new receipts.
                frame['post_state']={k:v for k,v in view.items() if k not in ('resolutions','actions')}
                # Older receipt identities remain available, while complete
                # result snapshots are bounded independently of log length.
                for i,r in enumerate(self.resolutions):
                    if i<len(self.resolutions)-15 and r.post_state is not None:
                        self.resolutions[i]=replace(r,post_state=None)
            self.resolutions.append(Resolution(at_revision=self.revision+1,complete=complete,log_position=self.power_lines,
                **{**frame,'zone_transitions':tuple(frame.get('zone_transitions',())),
                   'entity_changes':tuple(frame.get('entity_changes',()))}))
            self.stats["resolved_action_blocks" if complete else "partial_action_blocks"]+=1
        if not complete:self.stats["unterminated_blocks"]+=1

    def _abandon_blocks(self):
        while self.block_stack:self._finish_block(self.block_stack.pop(),False)

    def _block_action(self,body):
        bm=re.match(r"BLOCK_START BlockType=(\w+) Entity=(.*?) EffectCardId=.*? Target=(.*?)(?: SubOption=(-?\d+)|$)",body)
        if not bm or self.mode!="GT_BATTLEGROUNDS":return None
        if self.game_entity in self.entities and self.entities[self.game_entity].tags.get("BACON_IN_COMBAT_PHASE")=="1":return None
        block,ref,target_ref,sid=bm.groups()
        eid=self.resolve(ref)
        target=self.resolve(target_ref) if target_ref!="0" else None
        entity=self.entities.get(eid)
        if not entity:return None
        c=entity.card();cid=c.card_id.lower();kind=None;scope=None
        if block=="MOVE_MINION" and c.card_type=="MINION":
            kind=Kind.REORDER;scope="board" if c.controller==self.local_player_id else "shop"
        elif block=="PLAY":
            if c.controller==self.local_player_id:
                if "dragbuy" in cid:kind=Kind.BUY;eid,target=target,None
                elif "dragsell" in cid:kind=Kind.SELL;eid,target=target,None
                elif "techup" in cid:kind=Kind.UPGRADE
                elif "reroll_button" in cid:kind=Kind.REFRESH
                elif "lockall" in cid:kind=Kind.FREEZE
                elif c.zone=="HAND":kind=Kind.PLAY
                elif c.card_type=="HERO_POWER":kind=Kind.HERO_POWER
                elif c.card_type=="GAME_MODE_BUTTON" or c.tags.get("INTERACTABLE_OBJECT")=="1":kind=Kind.ACTIVATE
        return dict(kind=kind,source=eid,target=target,sub_option_id=integer(sid) if sid not in (None,"-1") else None,
                    scope=scope,block_type=block,started_at_revision=self.revision+1,started_at_log_position=self.power_lines) if kind else None

    def entity(self, eid: int) -> Entity:
        return self.entities.setdefault(eid, Entity(eid))

    def resolve(self, text: str) -> int | None:
        ref = REF.search(text)
        if ref:
            name, eid, zone, pos, cid, pid = ref.groups()
            e = self.entity(int(eid))
            e.card_id = e.card_id or cid
            if name and (not e.name or (cid and cid == e.card_id)):
                e.name = name
            # References often retain OLD values after TAG_CHANGE. Only explicit
            # tags may overwrite known state, otherwise removed combat copies
            # get resurrected by the next ATK/HEALTH update.
            for tag, value in (("ZONE",zone),("ZONE_POSITION",pos),("CONTROLLER",pid)):
                e.tags.setdefault(tag,value)
            self._infer_local(e)
            return e.id
        if text.isdigit():
            return int(text)
        if text == "GameEntity":
            return self.game_entity
        return self.aliases.get(text)

    def _infer_local(self, entity: Entity):
        # UI controller entities exist on the friendly side. Never infer from
        # opponent hands, turn ownership or a hard-coded player number.
        cid = entity.card_id.lower()
        if self.forced_player is None and ("reroll_button" in cid or "techup" in cid):
            pid = integer(entity.tags.get("CONTROLLER"))
            if pid is not None:
                self.local_player_id = pid

    def feed(self, line: str, *, source=None):
        self.stats["lines"] += 1
        if source!='scene' and 'LoadingScreen.' not in line:self.power_lines+=1
        m = LINE.match(line.strip())
        if not m:
            return
        method, body = m.group("method"), m.group("body").strip()
        self.client_objects.feed(method,body,self.power_lines)
        # These are client playback/submission evidence, NOT duplicate game
        # state. They must survive even when GameState is the semantic source.
        if method=='PowerProcessor.PrepareHistoryForCurrentTaskList':
            task=re.search(r'm_currentTaskList=(\d+)',body)
            if task:self.client_task_list=int(task[1])
            return
        if method=='PowerProcessor.EndCurrentTaskList':
            task=re.search(r'm_currentTaskList=(\d+)',body)
            if task and self.client_task_list==int(task[1]):
                self.client_task_list=None;self.client_playback_position=self.power_lines
            return
        if method in ('GameState.SendOption','GameState.SendChoices'):
            self.client_submission_position=self.power_lines
            return
        if method in ('ChoiceCardMgr.WaitThenShowChoices','ChoiceCardMgr.WaitThenHideChoicesFromPacket'):
            event=re.match(r'id=(\d+) (WAIT for taskList \d+|BEGIN|END WAIT)',body)
            if event:
                # BEGIN仅表示协程开始；21:36实战在其后3秒才显示面板。
                # 展示流程交由当前画面证明；隐藏流程不得误标为已显示。
                showing=method=='ChoiceCardMgr.WaitThenShowChoices'
                state=('waiting' if event[2].startswith('WAIT') else 'unknown') if showing else 'hidden'
                self.choice_display[int(event[1])]=state
                self.revision+=1;self.stats['choice_display_'+state]+=1
            return
        if method.startswith("LoadingScreen."):
            scene = re.search(r"currMode=(\w+)", body)
            if scene:
                self.scene = scene[1]
                self.revision += 1
            return
        if "DebugPrintGame" in method:
            if body.startswith("GameType="):
                self.mode = body.split("=", 1)[1]
            elif body.startswith("BuildNumber="):
                self.build = body.split("=", 1)[1]
            else:
                p = re.match(r"PlayerID=(\d+), PlayerName=(.*)", body)
                if p and int(p[1]) in self.players:
                    eid = self.players[int(p[1])]
                    self.aliases[p[2]] = eid
                    self.entity(eid).tags.update(self.pending_alias_tags.pop(p[2], {}))
            self.revision += 1
            return
        if "DebugPrintOptions" in method:
            if re.fullmatch(r"id=\d+",body):self._abandon_blocks()
            self._options(body)
            self.revision += 1
            return
        if "DebugPrintEntityChoices" in method:
            self._choices(body)
            self.revision += 1
            return
        if "DebugPrintEntitiesChosen" in method or "DebugPrintSendChoices" in method:
            cid = re.match(r"id=(\d+)", body)
            if 'DebugPrintEntitiesChosen' in method:
                if cid:
                    self.chosen_context=(int(cid[1]),self.revision+1);self.chosen_log_position=self.power_lines
                    self.chosen_rows={}
                    count=re.search(r'EntitiesCount=(\d+)',body)
                    self.chosen_count=int(count[1]) if count else None
                    owner=re.search(r'PlayerId=(\d+)',body)
                    ref=re.search(r'Player=(.*?) EntitiesCount=',body)
                    entity=self.entities.get(self.resolve(ref[1])) if ref else None
                    self.chosen_owner=int(owner[1]) if owner else integer(entity.tags.get('PLAYER_ID')) if entity else None
                elif self.chosen_context and (chosen:=re.match(r'Entities\[\d+\]=(.*)',body)):
                    eid=self.resolve(chosen[1]);choice_id,started=self.chosen_context
                    index=int(re.search(r'\[(\d+)\]',body)[1])
                    if eid is not None and index not in self.chosen_rows:
                        self.chosen_rows[index]=eid
                        self.resolutions.append(Resolution(self.revision+1,Kind.CHOOSE,eid,origin='server_choices',block_type='CHOICE:'+str(choice_id),started_at_revision=started,log_position=self.power_lines,started_at_log_position=self.chosen_log_position,choice_owner=self.chosen_owner))
                        self.revision+=1
            if 'DebugPrintEntitiesChosen' in method and cid and self.choice and self.choice["id"] == int(cid[1]) and (self.chosen_owner is None or self.chosen_owner==self.choice['owner']):
                self.choice = None
                self.choice_rows.clear()
                self.revision += 1
            return
        if not method.endswith(".DebugPrintPower"):
            return
        provider = method.split(".")[0]
        if body == "CREATE_GAME":
            if self.power_provider and provider != self.power_provider:
                return
            self.power_provider = provider
            self.entities.clear()
            self.players.clear()
            self.aliases.clear()
            self.pending_alias_tags.clear()
            self.options.clear()
            self.options_pending=False
            self.current_option = self.current_sub_option = None
            self.choice = None
            self.choice_rows.clear()
            self.chosen_context=None
            self.chosen_rows={};self.chosen_count=None;self.chosen_owner=None
            self.projection_invalid_reason=None
            self.choice_display.clear()
            self.client_task_list=None;self.client_submission_position=None;self.client_playback_position=None
            self.game_entity = self.active_entity = None
            self.mode = "unknown"
            self.local_player_id = self.forced_player
            self.generation += 1
            self.block_stack.clear();self.resolutions.clear()
            self.stats["snapshots"] += 1
        elif self.power_provider and provider != self.power_provider:
            return  # GameState projection and animation TaskList are duplicates.
        elif body=='RESET_GAME':
            self.client_task_list=None;self.client_submission_position=None;self.client_playback_position=None
            self._abandon_blocks()
            self.options.clear();self.choice=None;self.choice_rows.clear()
            self.current_option=self.current_sub_option=self.active_entity=None
            self.projection_invalid_reason='reset_requires_fresh_create_game'
            self.stats['unhandled_state_reset']+=1
        elif body.startswith("BLOCK_START "):
            raw_body=line.split(" - ",1)[-1]
            indent=len(raw_body)-len(raw_body.lstrip())
            while self.block_stack and self.block_stack[-1]["indent"]>=indent:
                self._finish_block(self.block_stack.pop(),False)
            frame=self._block_action(body) if not self.block_stack else None
            self.block_stack.append(dict(indent=indent,action=frame))
            self.active_entity=None
        elif body=="BLOCK_END":
            raw_body=line.split(" - ",1)[-1]
            indent=len(raw_body)-len(raw_body.lstrip())
            while self.block_stack and self.block_stack[-1]["indent"]>indent:
                self._finish_block(self.block_stack.pop(),False)
            if self.block_stack and self.block_stack[-1]["indent"]==indent:
                self._finish_block(self.block_stack.pop(),True)
            else:self.stats["unmatched_block_end"]+=1
            self.active_entity=None
        elif gm := re.match(r"GameEntity EntityID=(\d+)", body):
            self.game_entity = self.active_entity = int(gm[1])
            self.entity(self.active_entity).tags["CARDTYPE"] = "GAME"
        elif pm := re.match(r"Player EntityID=(\d+) PlayerID=(\d+)", body):
            eid, pid = int(pm[1]), int(pm[2])
            self.active_entity = eid
            self.players[pid] = eid
            self.entity(eid).tags.update(CARDTYPE="PLAYER", PLAYER_ID=str(pid), CONTROLLER=str(pid))
        elif fm := re.match(r"FULL_ENTITY - Creating ID=(\d+) CardID=(\w*)", body):
            self.active_entity = int(fm[1])
            # FULL_ENTITY is a complete snapshot, unlike SHOW/CHANGE_ENTITY.
            # Recreated ids must not inherit stats/keywords absent from it.
            self.entities[self.active_entity] = Entity(self.active_entity,fm[2])
        elif body.startswith(("FULL_ENTITY - Updating", "SHOW_ENTITY", "CHANGE_ENTITY")):
            numeric = re.search(r"\bEntity=(\d+)\b",body)
            full=body.startswith('FULL_ENTITY - Updating')
            reference=re.sub(r'^FULL_ENTITY - Updating\s+(?:Entity=)?','',body).split(' CardID=',1)[0] if full else body
            self.active_entity = self.resolve(numeric[1] if numeric else reference)
            if self.active_entity is not None:
                if full:self.entity(self.active_entity).tags.clear()
                cm = re.search(r"CardID=(\w*)", body)
                if cm:
                    self.entity(self.active_entity).card_id = cm[1]
        elif tc := re.match(r"(?:TAG_CHANGE|HIDE_ENTITY -) Entity=(.*?) tag=(\w+) value=([^\s]+)", body):
            ref, tag, value = tc.groups()
            eid = self.resolve(ref)
            if eid is not None:
                entity=self.entity(eid);old=entity.tags.get(tag)
                # 磁力可叠加到旧附魔；保留本处理根内的目标关联及附魔增量，不能只找新实体。
                magnetic_progress=(entity.tags.get('MAGNETIC')=='1' and
                    tag in ('TAG_SCRIPT_DATA_NUM_1','TAG_SCRIPT_DATA_NUM_2'))
                # 冻结按钮在空店也可能更新自身状态；保留同源处理内的变化，
                # 外层结束缺失时仍可独立核验“已处理”，不能伪造complete。
                freeze_progress=tag=='TAG_SCRIPT_DATA_NUM_1' and any(
                    f['action'] and f['action']['kind']==Kind.FREEZE and f['action']['source']==eid
                    for f in self.block_stack)
                if old!=value and (magnetic_progress or freeze_progress or tag in ('ZONE','ZONE_POSITION','CONTROLLER','CARDTYPE','PREMIUM','COPIED_FROM_ENTITY_ID')):
                    for frame in self.block_stack:
                        action=frame['action']
                        if action:
                            if tag=='ZONE_POSITION' and action['kind']!=Kind.REORDER:continue
                            changes=action.setdefault('entity_changes',[])
                            if len(changes)<512:changes.append((eid,tag,old or '',value))
                if tag=='ZONE' and old!=value:
                    for frame in self.block_stack:
                        action=frame['action']
                        if action and action['source']==eid:
                            transitions=action.setdefault('zone_transitions',[])
                            if len(transitions)<16:transitions.append((eid,NUM_ZONES.get(integer(old),old or ''),NUM_ZONES.get(integer(value),value)))
                referenced=any(eid==o['entity'] or eid in o['targets'] or any(eid==v['entity'] or eid in v['targets'] for v in o.get('sub_options',())) for o in self.options)
                card=entity.card()
                own=card.controller==self.local_player_id and card.card_type in HAND_TYPES+('HERO_POWER','GAME_MODE_BUTTON')
                resource=tag in ('RESOURCES','RESOURCES_USED','TEMP_RESOURCES') and eid==self.players.get(self.local_player_id)
                relevant=(tag in ('ZONE','CONTROLLER') and (referenced or own)) or resource or (tag=='BACON_IN_COMBAT_PHASE' and eid==self.game_entity)
                entity.tags[tag] = value
                self._infer_local(entity)
                if old!=value and relevant:
                    self.options_pending=True
                    self.options.clear()
                    self.current_option = None
                    self.current_sub_option = None
            else:
                self.pending_alias_tags.setdefault(ref, {})[tag] = value
                self.stats["unresolved_alias_updates"] += 1
            self.active_entity = None
        elif tm := TAG.fullmatch(body):
            if self.active_entity is not None:
                e = self.entity(self.active_entity)
                e.tags[tm[1]] = tm[2]
                self._infer_local(e)
        else:
            self.active_entity = None
        self.revision += 1

    def _options(self, body: str):
        if re.fullmatch(r"id=\d+", body):
            self.options_pending=False
            self.options.clear()
            self.current_option = None
            self.current_sub_option = None
        elif om := re.match(r"option (\d+) type=(\w+) mainEntity=(.*?) error=(\w+)", body):
            self.current_option = {"id": int(om[1]), "type": om[2], "entity": self.resolve(om[3]),
                                   "error": om[4], "targets": [], "has_targets": False}
            self.options.append(self.current_option)
            self.current_sub_option = None
        elif sm := re.match(r"subOption (\d+) entity=(.*?) error=(\w+)", body):
            if self.current_option:
                self.current_sub_option = {"id":int(sm[1]), "entity":self.resolve(sm[2]),
                                           "error":sm[3], "targets":[], "has_targets":False}
                self.current_option.setdefault("sub_options",[]).append(self.current_sub_option)
        elif tm := re.match(r"target \d+ entity=(.*?) error=(\w+)", body):
            recipient = self.current_sub_option if self.current_sub_option is not None else self.current_option
            if recipient:
                recipient["has_targets"] = True
            if recipient and tm[2] == "NONE":
                eid = self.resolve(tm[1])
                if eid is not None:
                    recipient["targets"].append(eid)

    def _choices(self, body: str):
        cm = re.match(r"id=(\d+) (?:Player=(.*?) TaskList=\d* |PlayerId=(\d+) )ChoiceType=(\w+) CountMin=(\d+) CountMax=(\d+)", body)
        if cm:
            self.choice = {"id": int(cm[1]), "type": cm[4], "min": int(cm[5]), "max": int(cm[6]), "owner": None}
            task=re.search(r'TaskList=(\d+)',body)
            self.choice.update(source_entity=None,task_list=int(task[1]) if task else None)
            if cm[3]:
                self.choice["owner"] = int(cm[3])
            elif (eid := self.resolve(cm[2])) in self.entities:
                self.choice["owner"] = integer(self.entities[eid].tags.get("PLAYER_ID"))
            self.choice_rows.clear()
        elif body.startswith('Source=') and self.choice:
            self.choice['source_entity']=self.resolve(body.split('=',1)[1])
        elif em := re.match(r"Entities\[(\d+)\]=(.*)", body):
            eid = self.resolve(em[2])
            if self.choice and eid is not None:
                self.choice_rows[int(em[1])] = eid
                if (self.local_player_id is None and self.mode == "GT_BATTLEGROUNDS" and
                    self.choice["type"] == "MULLIGAN" and self.choice["owner"] is not None and
                    integer(self.entities[eid].tags.get("CONTROLLER")) == self.choice["owner"]):
                    self.local_player_id = self.choice["owner"]

    def observe(self, *, source_updated_at=None) -> Observation:
        pid = self.local_player_id
        player = self.entities.get(self.players.get(pid))
        ptags = player.tags if player else {}
        gt = self.entities[self.game_entity].tags if self.game_entity in self.entities else {}
        needed=set(self.choice_rows.values())|{integer(ptags.get('HERO_ENTITY'))}
        reward_ids={integer(self.entities[eid].tags.get('TAG_SCRIPT_DATA_ENT_1')) for eid in self.choice_rows.values()
                    if eid in self.entities and self.entities[eid].tags.get('QUEST')=='1'}
        needed.update(reward_ids)
        for option in self.options:
            for row in (option,*option.get('sub_options',())):
                needed.add(row['entity']);needed.update(row['targets'])
        # Keep archived entities for future log references without rebuilding
        # every dead/combat copy into a Card on each action/receipt snapshot.
        cards = {eid:e.card() for eid,e in self.entities.items() if eid in needed or
            NUM_ZONES.get(integer(e.tags.get('ZONE')),e.tags.get('ZONE')) in ('PLAY','HAND')}
        # Dark Gift identity: the server links minion -> effect entity with
        # DARK_GIFT_ENTITY, and that entity keeps its card id after it leaves
        # PLAY. Publish the linked id on the visible card so a reconstructor
        # can apply the real gift instead of guessing from ATTACHED/name
        # suffixes. No hidden entity is exposed: only the id of a gift that
        # the client already displayed on this owned minion.
        from dataclasses import replace as _replace
        for eid,card in tuple(cards.items()):
            if card.card_type!='MINION' or card.tags.get('HAS_DARK_GIFT')!='1':continue
            gift=self.entities.get(integer(card.tags.get('DARK_GIFT_ENTITY')))
            if gift is None or not gift.card_id or gift.card_id==card.card_id:continue
            cards[eid]=_replace(card,tags=FrozenTags({**card.tags,'DARK_GIFT_CARD_ID':gift.card_id}))
        hero = cards.get(integer(ptags.get("HERO_ENTITY")))
        bob_ids = {integer(e.tags.get('CONTROLLER')) for e in self.entities.values() if 'baconshopbob' in e.card_id.lower()}
        sort = lambda cs: tuple(sorted(cs, key=lambda c: (c.position, c.entity_id)))
        hand = sort(c for c in cards.values() if c.controller == pid and pid is not None and c.zone == "HAND" and c.card_type in HAND_TYPES)
        board = sort(c for c in cards.values() if c.controller == pid and pid is not None and c.zone == "PLAY" and c.card_type == "MINION")
        shop = sort(c for c in cards.values() if c.controller in bob_ids and c.controller != pid and c.zone == "PLAY" and c.card_type in HAND_TYPES)
        is_bg = self.mode == "GT_BATTLEGROUNDS"
        phase = Phase.UNKNOWN
        if is_bg and pid is not None and player:
            # A newly created lobby can have a player/hero while still waiting
            # for the other players. Absence of combat is not recruit evidence.
            phase = Phase.COMBAT if gt.get("BACON_IN_COMBAT_PHASE") == "1" else Phase.RECRUIT if 'RESOURCES' in ptags else Phase.UNKNOWN
        if gt.get("STATE") in ("COMPLETE", "3") or ptags.get("PLAYSTATE") in ("WON", "LOST", "TIED", "CONCEDED", "4", "5", "6", "8"):
            phase = Phase.GAME_OVER
        if self.scene and self.scene not in ("GAMEPLAY", "INVALID"):
            phase = Phase.MENU
        warnings = []
        if self.projection_invalid_reason:warnings.append('state_health:'+self.projection_invalid_reason)
        if self.chosen_count is not None and len(self.chosen_rows)<self.chosen_count:
            warnings.append('state_health:choice_receipt_incomplete')
        if not is_bg:
            warnings.append("solo_battlegrounds_mode_not_confirmed")
        if pid is None:
            warnings.append("local_player_unknown: supply --player-id or wait for friendly control entities")
        gold = None
        if "RESOURCES" in ptags:
            gold = max(0, integer(ptags.get("RESOURCES"), 0) + integer(ptags.get("TEMP_RESOURCES"), 0) - integer(ptags.get("RESOURCES_USED"), 0))
        if gold is None:
            warnings.append("gold_unknown")
        actions = []
        choices = ()
        interaction = None
        if self.choice and self.choice["owner"] == pid and self.choice_rows and is_bg and phase not in (Phase.GAME_OVER,Phase.MENU):
            phase = Phase.CHOICE
            from dataclasses import replace
            # EntityChoices order is authoritative even when SETASIDE entities
            # have ZONE_POSITION=0 or belong to a different zone's old slots.
            choices = tuple(replace(cards[eid],position=i+1) for i,(_,eid) in enumerate(sorted(self.choice_rows.items())))
            interaction = InteractionContext('choice',self.choice['id'],self.choice['min'],self.choice['max'],
                protocol='server_single' if self.choice['min']==self.choice['max']==1 else 'unknown',
                source_entity=self.choice.get('source_entity'),task_list=self.choice.get('task_list'),
                owner=self.choice.get('owner'),choice_type=self.choice['type'],
                display_state=self.choice_display.get(self.choice['id'],'unknown'))
            if interaction.display_state in ('waiting','hidden'):warnings.append('choice_client_'+interaction.display_state)
            if self.choice['max']==1:
                actions = [Action(Kind.CHOOSE, c.entity_id, choice_id=self.choice["id"], evidence="server_choices") for c in choices]
            else:
                actions=[Action(Kind.SELECT,c.entity_id,choice_id=self.choice['id'],evidence='server_choices') for c in choices]
                warnings.append("multi_select_choice_requires_adapter")
        elif phase == Phase.RECRUIT:
            actions = self._actions(cards, pid, hand, board, shop)
        if self.projection_invalid_reason:actions=[]
        upgrade = next((c.cost for c in cards.values() if c.controller == pid and c.zone == "PLAY" and "techup" in c.card_id.lower()), None)
        return Observation(revision=self.revision, game_id=f"power-{self.generation}", mode=self.mode,
                           phase=phase, local_player_id=pid, turn=integer(ptags.get("TURN")), gold=gold,
                           tavern_tier=integer(ptags.get("PLAYER_TECH_LEVEL")), upgrade_cost=upgrade,
                           hero=hero, hand=hand, board=board, shop=shop, choices=choices,
                           branches=tuple(cards[eid] for eid in dict.fromkeys(v['entity'] for opt in self.options for v in opt.get('sub_options',())) if eid in cards),
                           choice_rewards=tuple(cards[eid] for eid in sorted(e for e in reward_ids if e is not None) if eid in cards and cards[eid].card_type=='BATTLEGROUND_QUEST_REWARD'),
                           actions=tuple(actions), warnings=tuple(warnings), captured_at=time.time(),
                           source_updated_at=source_updated_at, build=self.build, scene=self.scene,resolutions=tuple(self.resolutions),interaction=interaction,
                           controls=tuple(c for c in cards.values() if c.controller==pid and c.zone=='PLAY' and c.card_type!='MINION'),log_position=self.power_lines,
                           client_task_list=self.client_task_list,client_submission_position=self.client_submission_position,
                           client_playback_position=self.client_playback_position,
                           options_pending=self.options_pending,
                           ready=not any(w.startswith(('state_health:','choice_client_')) for w in warnings))

    def _actions(self, cards, pid, hand, board, shop):
        result = {}
        hands, shops = {c.entity_id for c in hand}, {c.entity_id for c in shop}
        for opt in self.options:
            if opt["error"] != "NONE":
                continue
            c = cards.get(opt["entity"])
            if c is None:
                continue
            cid = c.card_id.lower()
            targets = opt["targets"]
            candidates = []
            # A plain POWER on a visible minion also represents repositioning.
            # Positions are inferred by this adapter, not enumerated in Options.
            move_group = board if c.entity_id in {b.entity_id for b in board} else shop if c.entity_id in shops else ()
            plain_move=opt['type']=='POWER' and c.card_type=='MINION' and move_group and not opt.get('has_targets') and not opt.get('sub_options') and opt is next(
                (o for o in self.options if o['entity']==c.entity_id and o['type']=='POWER' and not o.get('has_targets') and not o.get('sub_options')),None)
            if plain_move:
                ids = [b.entity_id for b in move_group if b.card_type == "MINION"]
                for position in range(1,len(ids)+1):
                    order = [eid for eid in ids if eid != c.entity_id]
                    order.insert(position-1,c.entity_id)
                    candidates.append(Action(Kind.REORDER,c.entity_id,option_id=opt["id"],order=tuple(order),
                                             position=position,scope="board" if move_group is board else "shop",evidence="bg_move_adapter"))
            if c.controller == pid and "dragsell" in cid:
                candidates = [Action(Kind.SELL, t, option_id=opt["id"]) for t in targets if t in {b.entity_id for b in board}]
            elif c.controller == pid and "dragbuy" in cid:
                candidates = [Action(Kind.BUY, t, option_id=opt["id"],resource_cost=c.cost,authorization_source=c.entity_id) for t in targets if t in shops]
            elif c.entity_id in shops:
                # Main shop-card POWER may mean inspect/activate, even at zero
                # gold. Only DragBuy + legal target proves a purchase is legal.
                pass
            else:
                kind = None
                if c.controller == pid:
                    if "techup" in cid:
                        kind = Kind.UPGRADE
                    elif "reroll_button" in cid:
                        kind = Kind.REFRESH
                    elif "lockall" in cid:
                        kind = Kind.FREEZE
                    elif c.entity_id in hands:
                        kind = Kind.PLAY
                    elif c.card_type == "HERO_POWER":
                        kind = Kind.HERO_POWER
                    elif c.card_type == "GAME_MODE_BUTTON" or (c.zone == "PLAY" and c.tags.get("INTERACTABLE_OBJECT") == "1" and not plain_move):
                        kind = Kind.ACTIVATE
                if kind:
                    variants = opt.get("sub_options") or [opt]
                    for variant in variants:
                        if variant["error"] != "NONE":
                            continue
                        sid = variant["id"] if "sub_options" in opt else None
                        branch=None
                        if sid is not None and all(v['entity'] is not None for v in opt['sub_options']):
                            branch=InteractionSpec(stages=tuple(InteractionStage('branch','candidate',v['entity']) for v in opt['sub_options'])+
                                (InteractionStage('branch','choose',variant['entity']),),mandatory=True)
                        ts = variant["targets"]
                        if ts:
                            for t in ts:
                                spec=branch;target=cards.get(t)
                                from .magnetic import magnetic,mechanical
                                if kind==Kind.PLAY and magnetic(c) and c.tags.get('BATTLECRY')!='1' and sid is None and target and target.controller==pid and target.zone=='PLAY' and mechanical(target):
                                    spec=InteractionSpec(placement='magnetize',mandatory=True)
                                candidates.append(Action(kind,c.entity_id,target=t,option_id=opt['id'],sub_option_id=sid,interaction=spec))
                        elif not variant.get("has_targets"):
                            candidates.append(Action(kind,c.entity_id,option_id=opt["id"],sub_option_id=sid,interaction=branch))
            for a in candidates:
                result.setdefault(a.key, a)
        return list(result.values())
