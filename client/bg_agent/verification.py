"""Conservative action-specific postconditions, independent of UI coordinates.

CONFIRMED proves the stated observable postcondition. RESOLVED only proves a
matching server block finished, not that every effect or chosen target benefited
the player. PENDING/CONTEXT_CHANGED must never trigger blind retry.
"""
from dataclasses import dataclass
from enum import StrEnum

from .model import Kind,Observation,Phase


class Status(StrEnum):
    CONFIRMED="confirmed"
    RESOLVED="resolved"
    PENDING="pending"
    CONTEXT_CHANGED="context_changed"
    UNSUPPORTED="unsupported"


ABILITY_PROGRESS_TAGS=('BACON_HERO_POWER_ACTIVATED','TAG_SCRIPT_DATA_NUM_1','TAG_SCRIPT_DATA_NUM_2',
                       'SCRIPT_DATA_NUM_1','SCRIPT_DATA_NUM_2','INTERACTABLE_OBJECT_NUM_USES')


@dataclass(frozen=True)
class Outcome:
    status: Status
    reason: str
    receipt_revision: int | None = None
    proofs: tuple[str, ...] = ()


def action_progress(action,before,after,outcome):
    """报告证据的具体强度，不把本地输入或根结束等同所有动画结束。"""
    receipts=[r for r in after.resolutions if r.at_revision==outcome.receipt_revision
        and r.kind==action.kind and r.source==action.source]
    return dict(client_submission_observed=bool(after.client_submission_position is not None
        and (before.client_submission_position is None or after.client_submission_position>before.client_submission_position)),
        server_response_observed=bool(receipts),root_finished=any(r.complete for r in receipts),
        necessary_postcondition_observed=outcome.status==Status.CONFIRMED or accepts_processed(action,before,after,outcome),
        forced_interaction_open=after.interaction is not None or after.phase==Phase.CHOICE,
        client_playback_busy=after.client_task_list is not None,next_action_ready='requires_new_action_evidence')


def accepts_processed(action, before, after, outcome):
    """Shared strict continuation contract; receipt text is not a proof type."""
    if outcome.status != Status.RESOLVED:return False
    proofs=set(outcome.proofs)
    if not {'matching_receipt','mandatory_ui_closed'}<=proofs:return False
    if action.kind==Kind.CHOOSE:
        selected=next((c for c in before.choices if c.entity_id==action.source),None)
        return bool(selected and selected.card_type not in ('HERO','BATTLEGROUND_TRINKET') and 'request_closed' in proofs)
    if action.kind==Kind.PLAY:return 'source_consumed' in proofs or {'ordinary_summon_observed','placement_deviated'}<=proofs
    if action.kind==Kind.FREEZE:
        return not before.shop and not after.shop and 'empty_shop_freeze_processed' in proofs and (
            'root_finished' in proofs or {'freeze_control_state_changed','matching_submission_before_root',
                'client_playback_completed'}<=proofs)
    if action.kind==Kind.REORDER:
        return action.scope=='board' and {'board_permutation_observed','placement_deviated'}<=proofs
    if action.kind in (Kind.HERO_POWER,Kind.ACTIVATE):
        return bool(proofs & {'ability_paid','ability_exhausted','ability_progress','observable_effect'})
    return False


def consumed_purchase_replacement(action,before,after,receipt):
    """Identity-bound proof for a purchased source merged before observation.

    Missing outer BLOCK_END must remain incomplete. It does not erase the
    independently observed ownership/hand transfer and its causal replacement.
    Only COPY edges in this matching root may link a NEW owned result to the
    exact purchased source. Card names, premium flags and gold alone are not
    evidence. The provider still gates the next input on transaction/UI state.
    """
    source=action.source;owner=str(before.local_player_id)
    changes=receipt.entity_changes
    transferred=any(eid==source and tag=='CONTROLLER' and old!=owner and new==owner
                    for eid,tag,old,new in changes)
    zones=[(old,new) for eid,old,new in receipt.zone_transitions if eid==source]
    entered=next((i for i,event in enumerate(zones) if event==('PLAY','HAND')),None)
    consumed=entered is not None and any(old=='HAND' and new in ('SETASIDE','REMOVEDFROMGAME','GRAVEYARD')
                                        for old,new in zones[entered+1:])
    if not transferred or not consumed:return None
    if source in {c.entity_id for c in after.hand+after.board+after.shop}:return None
    graph={}
    for eid,tag,old,new in changes:
        if tag=='COPIED_FROM_ENTITY_ID':
            try:parent=int(new)
            except (ValueError,TypeError):continue
            graph.setdefault(eid,set()).add(parent)
    old_owned={c.entity_id for c in before.hand+before.board}
    for card in after.hand+after.board:
        if card.entity_id in old_owned or card.controller!=before.local_player_id:continue
        pending=[card.entity_id];seen=set()
        while pending:
            eid=pending.pop()
            if eid in seen:continue
            seen.add(eid)
            if eid==source:return card.entity_id
            pending.extend(graph.get(eid,()))
    return None


NO_EFFECT_KINDS=(Kind.BUY,Kind.SELL,Kind.PLAY,Kind.REORDER,Kind.UPGRADE,Kind.REFRESH,Kind.FREEZE,
                 Kind.HERO_POWER,Kind.ACTIVATE)


def no_effect_proven(action,before: Observation,after: Observation):
    """Prove that a dispatched input provably changed nothing.

    Returns (proven, reason). Only decisive, card-agnostic evidence counts: a
    bought shop entity is still offered, a sold minion is still on the board, the board
    order is unchanged, a paid ability did not charge. A matching receipt is
    never "no effect" - the server already started processing it.

    Forced interactions (choice/select/confirm/target/cancel) are never
    reconcilable: an open modal must be resolved, not skipped. This function
    authorises only "abandon this action and continue", never a resend.
    A hand card remaining unchanged cannot prove an unlogged client target
    stage is closed; PLAY must stop when its result cannot be established.
    """
    if action.kind not in NO_EFFECT_KINDS:return False,'kind_not_reconcilable'
    if before.phase!=Phase.RECRUIT:return False,'before_phase_not_recruit'
    if before.game_id!=after.game_id or before.mode!=after.mode:return False,'game_or_mode_changed'
    if after.phase not in (Phase.RECRUIT,Phase.COMBAT):return False,'after_phase_not_reconcilable'
    if after.interaction is not None:return False,'interaction_open'
    if action.sub_option_id is not None or (action.interaction and action.interaction.stages):
        # 客户端抉择面板可先于服务端请求出现；父牌仍在手中不等于没有打开模态界面。
        return False,'compound_interaction_unresolved'
    log_progress=before.log_position is not None and after.log_position is not None and after.log_position>before.log_position
    if after.revision<before.revision:return False,'revision_moved_backwards'
    if after.revision<=before.revision and not log_progress:return False,'no_new_observation'
    # A submitted option awaiting a server result is not an ignored click.
    # Never use unchanged tags or animation traffic to abandon that request.
    if (before.log_position is not None and after.client_submission_position is not None
            and after.client_submission_position>before.log_position):return False,'client_submission_observed'
    if after.client_task_list is not None:return False,'client_playback_busy'
    if before.turn is not None and after.turn is not None and after.turn<before.turn:return False,'turn_moved_backwards'
    if any(w.startswith('state_health:') for w in after.warnings):return False,'projection_unhealthy'
    for r in after.resolutions:
        if r.at_revision<=before.revision:continue
        if r.kind!=action.kind or r.source!=action.source:continue
        if action.target is not None and r.target is not None and r.target!=action.target:continue
        if action.scope is not None and r.scope is not None and r.scope!=action.scope:continue
        return False,'matching_receipt_observed'
    if action.kind in (Kind.HERO_POWER,Kind.ACTIVATE,Kind.UPGRADE,Kind.REFRESH,Kind.FREEZE):
        # Gold and the shop reset every turn, so these kinds only have decisive
        # evidence inside the same turn; a cross-turn claim would be a guess.
        if before.turn is None or after.turn!=before.turn:return False,'turn_changed_weak_evidence'
    if action.kind==Kind.PLAY:
        source=next((c for c in before.hand if c.entity_id==action.source),None)
        current=next((c for c in after.hand if c.entity_id==action.source),None)
        if source is None or current is None:return False,'source_not_still_in_hand'
        if current.zone!='HAND' or current.card_id!=source.card_id:return False,'source_identity_changed'
        # 原牌未离手不能排除拖中了邻牌、客户端已进入未记录的目标阶段。
        # 没有独立关闭证据时保持结果不明并停止，不能继续普通动作。
        return False,'hand_input_outcome_unproven'
    if action.kind==Kind.BUY:
        if action.source not in {c.entity_id for c in after.shop}:return False,'shop_source_gone'
        return True,'shop_source_still_offered'
    if action.kind==Kind.SELL:
        if action.source not in {c.entity_id for c in after.board}:return False,'board_source_gone'
        return True,'sold_entity_still_on_board'
    if action.kind==Kind.REORDER:
        if not before.board:return False,'empty_board'
        if tuple(c.entity_id for c in after.board)!=tuple(c.entity_id for c in before.board):
            return False,'board_order_changed'
        return True,'board_order_unchanged'
    if action.kind in (Kind.HERO_POWER,Kind.ACTIVATE):
        prior=next((c for c in before.controls+before.board if c.entity_id==action.source),None)
        current=next((c for c in after.controls+after.board if c.entity_id==action.source),None)
        if prior is None or current is None:return False,'control_not_observed'
        raw=prior.cost if action.kind==Kind.HERO_POWER or prior.card_type=='GAME_MODE_BUTTON' else prior.tags.get('INTERACTABLE_OBJECT_COST')
        try:cost=int(raw) if raw is not None else None
        except (TypeError,ValueError):cost=None
        if not cost or cost<=0:return False,'free_ability_not_provable'
        if before.gold is None or after.gold is None or after.gold!=before.gold:return False,'gold_changed'
        if any(prior.tags.get(key)!=current.tags.get(key) for key in ABILITY_PROGRESS_TAGS):
            return False,'ability_progress_observed'
        if prior.tags.get('EXHAUSTED')!=current.tags.get('EXHAUSTED'):return False,'ability_state_changed'
        old={c.entity_id:(c.attack,c.health) for c in before.board+before.hand}
        new={c.entity_id:(c.attack,c.health) for c in after.board+after.hand}
        if old!=new:return False,'board_or_hand_changed'
        return True,'paid_ability_not_charged'
    if action.kind==Kind.UPGRADE:
        if before.tavern_tier is None or after.tavern_tier is None:return False,'tier_unknown'
        if after.tavern_tier!=before.tavern_tier:return False,'tier_changed'
        if before.gold is not None and after.gold is not None and after.gold!=before.gold:return False,'gold_changed'
        return True,'tier_and_gold_unchanged'
    if action.kind==Kind.REFRESH:
        old=tuple(c.entity_id for c in before.shop)
        if not old or old!=tuple(c.entity_id for c in after.shop):return False,'shop_set_changed'
        return True,'shop_set_unchanged'
    if action.kind==Kind.FREEZE:
        old={c.entity_id:c.tags.get('FROZEN','0') for c in before.shop}
        new={c.entity_id:c.tags.get('FROZEN','0') for c in after.shop}
        if not old or old!=new:return False,'freeze_state_changed'
        return True,'freeze_state_unchanged'
    return False,'unsupported_kind'


def verify(action,before: Observation,after: Observation) -> Outcome:
    # Providers may attach explicit projection health warnings. Never confirm
    # a result from such a snapshot; ordinary adapter warnings remain separate.
    if any(w.startswith('state_health:') for w in after.warnings):
        return Outcome(Status.PENDING,'状态投影健康检查尚未通过')
    if action.kind==Kind.DISMISS and before.source=='hearthstone_live' and action.evidence=='verified_client_ui':
        if before.phase==Phase.GAME_OVER and after.phase==Phase.MENU and after.scene=='BACON':return Outcome(Status.CONFIRMED,'结算已关闭，单人酒馆菜单已由场景与画面确认')
        return Outcome(Status.PENDING,'等待结算关闭及单人酒馆菜单')
    if action.kind==Kind.ENTER and before.source=='hearthstone_live' and action.evidence=='verified_client_ui':
        if before.phase==Phase.MENU and after.phase in (Phase.CHOICE,Phase.RECRUIT) and after.mode=='GT_BATTLEGROUNDS' and after.revision>before.revision:
            return Outcome(Status.CONFIRMED,'开始后单人酒馆对局已建立并进入英雄选择或招募')
        return Outcome(Status.PENDING,'开始已发送，等待真实对局建立')
    if action.kind==Kind.END and before.source!='sandbox' and action.evidence=='verified_phase_wait':
        if (before.game_id,before.mode,before.local_player_id)!=(after.game_id,after.mode,after.local_player_id):return Outcome(Status.CONTEXT_CHANGED,'等待回合结束时对局身份已变化')
        if before.phase!=Phase.RECRUIT or before.interaction:return Outcome(Status.UNSUPPORTED,'当前阶段不能结束招募决策')
        if after.revision>before.revision and (after.phase==Phase.COMBAT or after.phase==Phase.RECRUIT and before.turn is not None and after.turn is not None and after.turn>before.turn):return Outcome(Status.CONFIRMED,'未发送结束按钮输入；日志已确认招募阶段结束')
        return Outcome(Status.PENDING,'等待日志切换至战斗或下一招募回合')
    if action.kind==Kind.CANCEL and before.source!='sandbox':
        from .client_protocol import cancellable_target
        if not cancellable_target(before,action):return Outcome(Status.UNSUPPORTED,'原请求不允许取消')
        if (before.game_id,before.turn,before.local_player_id,before.mode)!=(after.game_id,after.turn,after.local_player_id,after.mode):return Outcome(Status.CONTEXT_CHANGED,'取消期间请求所属回合已变化')
        if after.revision>before.revision and after.interaction is None and after.ui_revision>before.ui_revision:return Outcome(Status.CONFIRMED,'原可取消请求及客户端目标界面均已关闭')
        return Outcome(Status.PENDING,'等待原请求关闭及新的客户端退出证据')
    if action.kind==Kind.WAIT or (action.kind in (Kind.ENTER,Kind.DISMISS,Kind.END) and before.source!="sandbox"):
        return Outcome(Status.UNSUPPORTED,"此动作没有真实客户端结果适配器")
    if before.game_id!=after.game_id or before.mode!=after.mode:
        return Outcome(Status.CONTEXT_CHANGED,"对局或模式已变化，不能确认此前输入")
    if before.mode!="GT_BATTLEGROUNDS" and before.source!="sandbox":
        return Outcome(Status.UNSUPPORTED,"仅核验单人酒馆")
    ui_progress=action.kind==Kind.SELECT and before.interaction and before.interaction.protocol=='verified_multi_select' and after.ui_revision>before.ui_revision
    log_progress=before.log_position is not None and after.log_position is not None and after.log_position>before.log_position
    if after.revision<=before.revision and not ui_progress and not log_progress:
        return Outcome(Status.PENDING,"尚无更新观测")
    def newer(r):
        if before.log_position is not None and r.log_position is not None and r.started_at_log_position is not None:
            return r.log_position>before.log_position and r.started_at_log_position>before.log_position
        return r.at_revision>before.revision and (r.started_at_revision is None or r.started_at_revision>before.revision)
    from .magnetic import magnetic_context
    magnetic_play=magnetic_context(before,action)
    receipts=[r for r in after.resolutions if newer(r) and r.kind==action.kind
              and r.source==action.source and (r.target==action.target or magnetic_play and r.target is None) and r.sub_option_id==action.sub_option_id
              and (action.scope is None or r.scope==action.scope)
              and (r.origin=="server_power" or (r.origin=='server_choices' and action.kind==Kind.CHOOSE and r.block_type=='CHOICE:'+str(action.choice_id) and r.choice_owner in (None,before.local_player_id)) or (r.origin=="sandbox_ui" and before.source==after.source=="sandbox"))]
    # A stream resync can retain a partial root before a newer complete root
    # for the same source/target. Never let weaker old evidence hide the
    # complete matching result indefinitely (notably Magnetic attachment).
    receipt=max(receipts,key=lambda r:(r.complete,r.at_revision)) if receipts else None
    turn_changed=before.turn is not None and after.turn!=before.turn and not (action.kind==Kind.DISMISS and before.source==after.source=='sandbox' and after.phase==Phase.MENU)
    # Combat/result playback can remove or replace board entities within the
    # SAME turn. A complete, identity-bound root snapshot still proves the
    # preceding input; never rewrite current state with that historical board.
    phase_advanced=before.phase in (Phase.RECRUIT,Phase.CHOICE) and after.phase in (Phase.COMBAT,Phase.GAME_OVER)
    if turn_changed or phase_advanced:
        from dataclasses import replace
        for r in sorted(receipts,key=lambda r:(r.complete,r.at_revision),reverse=True):
            proof=r.post_state
            if r.complete and proof and proof.get('turn')==before.turn and proof.get('local_player_id')==before.local_player_id and proof.get('mode')==before.mode:
                snapshot=Observation.from_dict({**proof,'game_id':after.game_id,'revision':r.at_revision,
                    'resolutions':[],'actions':[]})
                outcome=verify(action,before,replace(snapshot,resolutions=(replace(r,post_state=None),)))
                if outcome.status==Status.CONFIRMED or accepts_processed(action,before,snapshot,outcome):return outcome
        if turn_changed:return Outcome(Status.CONTEXT_CHANGED,"回合变化且没有同回合完整处理快照，不能猜测此前输入结果")
    rev=receipt.at_revision if receipt else None
    def confirmed(reason,proofs=()):return Outcome(Status.CONFIRMED,reason+("（外层结束标记缺失，使用具体状态后置条件）" if receipt and not receipt.complete else ""),rev,proofs)
    if before.source==after.source=='sandbox' and receipt:
        if action.kind==Kind.ENTER and after.scene=='MATCHMAKING' and after.phase==Phase.UNKNOWN:
            return confirmed('受控开始操作已进入匹配等待')
        if action.kind==Kind.CONFIRM and before.interaction and after.interaction is None and action.selected==before.interaction.selected and action.choice_id==before.interaction.request_id:
            return confirmed('受控选择集合已提交且确认界面关闭')
        expected={Kind.ENTER:Phase.RECRUIT,Kind.END:Phase.COMBAT,Kind.DISMISS:Phase.MENU}.get(action.kind)
        if expected is not None and after.phase==expected:
            return confirmed('受控生命周期已进入请求的阶段')
    if action.kind==Kind.SELECT:
        selected=action.selected or ((action.source,) if action.source is not None else ())
        if ui_progress and after.interaction and after.interaction.protocol=='verified_multi_select' and before.interaction.request_id==after.interaction.request_id and after.interaction.selected==selected:
            return Outcome(Status.CONFIRMED,'当前请求的选择集合已由新的逐卡牌边框状态确认；尚未提交服务器')
        if receipt and selected and after.interaction and after.interaction.selected==selected:
            if before.interaction and before.interaction.request_id==after.interaction.request_id:
                return confirmed('当前选择请求的已选实体与动作一致')
        return Outcome(Status.PENDING,'选择状态或匹配回执尚未确认',rev)
    if action.kind==Kind.CONFIRM and before.interaction and before.interaction.protocol=='verified_multi_select':
        chosen={r.source for r in after.resolutions if r.origin=='server_choices' and r.kind==Kind.CHOOSE and r.block_type=='CHOICE:'+str(action.choice_id) and r.choice_owner in (None,before.local_player_id) and newer(r)}
        closed=not after.interaction or after.interaction.request_id!=before.interaction.request_id
        if closed and set(action.selected)==chosen:return Outcome(Status.CONFIRMED,'服务器已回执完整的选择集合，原多选请求已关闭')
        return Outcome(Status.PENDING,'等待服务器回执完整选择集合和请求关闭')
    if action.kind==Kind.TARGET:
        if receipt and before.interaction and before.interaction.mode=='target' and (not after.interaction or after.interaction.mode!='target'):
            return confirmed('匹配的目标提交已结束，目标选择界面关闭')
        return Outcome(Status.PENDING,'目标选择仍未确认',rev)
    if action.kind==Kind.CHOOSE:
        selected=next((c for c in before.choices if c.entity_id==action.source),None)
        closed_request=not after.choices or bool(before.interaction and after.interaction and before.interaction.request_id!=after.interaction.request_id)
        if selected and selected.card_type=='HERO' and closed_request and after.hero and after.hero.entity_id==selected.entity_id:
            return confirmed('所选英雄已成为本地英雄，原英雄选择请求已关闭')
        if selected and selected.card_type=='BATTLEGROUND_TRINKET' and closed_request:
            def active_count(obs):return sum(c.card_id==selected.card_id and c.controller==before.local_player_id and c.zone=='PLAY' for c in obs.controls)
            if active_count(after)>active_count(before):
                return confirmed('选择请求已关闭，所选饰品身份已出现在新的己方激活控件中')
        if selected and not after.choices:
            if selected.card_type=='BATTLEGROUND_TRINKET' and any(c.entity_id==selected.entity_id and c.controller==before.local_player_id and c.zone=='PLAY' for c in after.controls):
                return confirmed('所选饰品已成为己方激活控件，选择界面已关闭')
            if after.hero and after.hero.entity_id==selected.entity_id:
                return confirmed("所选英雄已成为本地英雄")
            old=sum(c.card_id==selected.card_id for c in before.hand)
            new=sum(c.card_id==selected.card_id for c in after.hand)
            if new>old and receipt:return confirmed("匹配选择回执、原请求关闭且所选卡牌数量增加")
        if receipt:
            closed=bool(before.interaction and (not after.interaction or after.interaction.request_id!=before.interaction.request_id))
            return Outcome(Status.RESOLVED,"匹配的选择提交已由回执确认；奖励效果另行核验",rev,
                ('matching_receipt',)+(('request_closed','mandatory_ui_closed') if closed else ()))
        return Outcome(Status.PENDING,"尚未取得所选实体对应的提交回执或明确选择结果")
    if not receipt:return Outcome(Status.PENDING,"没有新的、与本动作实体和目标匹配的处理回执")
    if action.kind==Kind.BUY:
        prior={c.entity_id for c in before.shop}
        owned={c.entity_id for c in after.hand+after.board if c.controller==before.local_player_id}
        if action.source in prior and action.source in owned and action.source not in {c.entity_id for c in after.shop}:
            return confirmed("商店目标已转移到己方手牌或场上")
        if action.source in prior and consumed_purchase_replacement(action,before,after,receipt) is not None:
            return confirmed('购买源已转移至己方手牌并被合并/消耗；同一处理的实体复制链证明新归属结果，不依赖卡名或伪造结束标记',
                ('matching_receipt','purchase_ownership_transfer','purchase_source_consumed','causal_replacement_observed'))
        if receipt.complete and action.source in prior and action.source not in {c.entity_id for c in after.shop}:
            # Buying a third copy can consume the purchased entity before the
            # next observation. Prove its actual ownership/zone transition in
            # this action, not a guessed relationship to a newly generated card.
            changes=[(tag,new) for eid,tag,old,new in receipt.entity_changes if eid==action.source]
            transferred=('CONTROLLER',str(before.local_player_id)) in changes
            entered=any(eid==action.source and new in ('HAND','PLAY') for eid,old,new in receipt.zone_transitions)
            if transferred and entered:
                return confirmed('购买源已在同一完整处理内转移至己方并进入手牌或场上；随后合并或消耗，不声称奖励身份')
    elif action.kind==Kind.SELL:
        if action.source in {c.entity_id for c in before.board} and action.source not in {c.entity_id for c in after.board}:
            return confirmed("指定出售随从已离开己方场面")
    elif action.kind==Kind.FREEZE:
        old={c.entity_id:c.tags.get('FROZEN','0') for c in before.shop}
        new={c.entity_id:c.tags.get('FROZEN','0') for c in after.shop}
        if old and old.keys()==new.keys() and old!=new and len(set(new.values()))==1:
            return confirmed('同一商店实体集合的冻结状态已一致切换')
        if not old and not new and receipt.complete and after.phase==Phase.RECRUIT and after.interaction is None:
            return Outcome(Status.RESOLVED,'空酒馆的匹配冻结处理已结束；没有商店卡牌状态可切换',rev,
                ('matching_receipt','mandatory_ui_closed','root_finished','empty_shop_freeze_processed'))
        if (not old and not new and after.phase==Phase.RECRUIT and after.interaction is None
                and not after.options_pending and after.client_task_list is None):
            prior=next((c for c in before.controls if c.entity_id==action.source),None)
            current=next((c for c in after.controls if c.entity_id==action.source),None)
            tag='TAG_SCRIPT_DATA_NUM_1'
            bound=bool(prior and current and prior.card_type==current.card_type=='GAME_MODE_BUTTON'
                and prior.controller==current.controller==before.local_player_id
                and prior.card_id==current.card_id and prior.zone==current.zone=='PLAY')
            changed=bool(bound and tag in prior.tags and tag in current.tags and prior.tags[tag]!=current.tags[tag]
                and (action.source,tag,prior.tags[tag],current.tags[tag]) in receipt.entity_changes)
            submitted=bool(before.log_position is not None and after.client_submission_position is not None
                and receipt.started_at_log_position is not None
                and before.log_position<after.client_submission_position<receipt.started_at_log_position)
            played=bool(receipt.started_at_log_position is not None and after.client_playback_position is not None
                and after.client_playback_position>receipt.started_at_log_position)
            if changed and submitted and played:
                return Outcome(Status.RESOLVED,'空酒馆冻结来源控件已在匹配处理内更新，客户端播放已结束；不伪造外层结束标记',rev,
                    ('matching_receipt','mandatory_ui_closed','empty_shop_freeze_processed',
                     'freeze_control_state_changed','matching_submission_before_root','client_playback_completed'))
    elif action.kind in (Kind.HERO_POWER,Kind.ACTIVATE):
        prior=next((c for c in before.controls+before.board if c.entity_id==action.source),None)
        current=next((c for c in after.controls+after.board if c.entity_id==action.source),None)
        closed=after.interaction is None and after.phase!=Phase.CHOICE
        if prior and closed:
            raw_cost=prior.cost if action.kind==Kind.HERO_POWER or prior.card_type=='GAME_MODE_BUTTON' else prior.tags.get('INTERACTABLE_OBJECT_COST')
            try:cost=int(raw_cost) if raw_cost is not None else None
            except (TypeError,ValueError):cost=None
            paid=cost is not None and cost>0 and before.gold is not None and after.gold is not None and after.gold<=before.gold-cost
            exhausted=current and prior.tags.get('EXHAUSTED')!='1' and current.tags.get('EXHAUSTED')=='1'
            progress=bool(current and any(k in prior.tags and k in current.tags and prior.tags[k]!=current.tags[k] for k in ABILITY_PROGRESS_TAGS))
            old={c.entity_id:c for c in before.board+before.hand}
            new={c.entity_id:c for c in after.board+after.hand}
            effect=bool(receipt.complete and (set(new)-set(old) or any(
                (old[e].attack,old[e].health)!=(new[e].attack,new[e].health) for e in old.keys()&new.keys()) or
                cost==0 and before.gold is not None and after.gold is not None and after.gold>before.gold))
            proofs=('matching_receipt','mandatory_ui_closed')+tuple(name for name,yes in (
                ('ability_paid',paid),('ability_exhausted',exhausted),('ability_progress',progress),('observable_effect',effect)) if yes)
            if paid or exhausted or progress or effect:
                return Outcome(Status.RESOLVED,'匹配主动能力已有费用、耗竭、进度或可观察效果证明，强制界面已关闭；不声称全部收益已核实',rev,proofs)
    elif action.kind==Kind.UPGRADE:
        if before.tavern_tier is not None and after.tavern_tier is not None and after.tavern_tier>before.tavern_tier:
            return confirmed("酒馆等级已增加")
    elif action.kind==Kind.REFRESH:
        if before.shop and after.shop and {c.entity_id for c in before.shop}!={c.entity_id for c in after.shop}:
            return confirmed("商店实体集合已变化")
        if not before.shop and after.shop and receipt.complete and after.phase==Phase.RECRUIT:
            # 买空酒馆后刷新也合法。不能要求旧集合非空，更不能因扣费就确认。
            # 复用上面的来源/日志游标/回合校验，要求本次刷新根结束且新货已出现。
            return confirmed('空酒馆已由本次匹配刷新处理补出新货',
                ('matching_receipt','refresh_root_finished','empty_shop_populated'))
    elif action.kind==Kind.REORDER:
        group=after.shop if action.scope=="shop" else after.board
        ids=tuple(c.entity_id for c in group if c.card_type=="MINION")
        if ids==action.order:return confirmed("指定区域顺序与换位请求一致")
        # Do not wait forever for the requested order after the client has
        # finished a different, fully observable move. Require an actual
        # source-bound MOVE_MINION root plus its position changes, not merely
        # some board change or generic PLAY receipt. Never label it success.
        old_ids=tuple(c.entity_id for c in before.board)
        changes=[new for eid,tag,old,new in receipt.entity_changes
                 if eid==action.source and tag=='ZONE_POSITION' and old!=new]
        placed=next((c for c in after.board if c.entity_id==action.source),None)
        snapshot=receipt.post_state or {}
        root_board=snapshot.get('board',[])
        root_bound=(snapshot.get('turn')==before.turn and snapshot.get('local_player_id')==before.local_player_id
            and snapshot.get('mode')==before.mode and snapshot.get('phase')==Phase.RECRUIT
            and [(c['entity_id'],c['position'],c['controller']) for c in root_board]
                ==[(c.entity_id,c.position,c.controller) for c in after.board])
        if (action.scope=='board' and receipt.complete and receipt.block_type=='MOVE_MINION'
            and receipt.scope=='board' and after.phase==Phase.RECRUIT and after.interaction is None
            and placed and (str(placed.position) in changes or root_bound) and set(ids)==set(old_ids)==set(action.order)
            and len(ids)==len(old_ids)==len(set(ids))
            and all(c.controller==before.local_player_id for c in after.board)
            and sorted(c.position for c in after.board)==list(range(1,len(ids)+1))):
            return Outcome(Status.RESOLVED,'换位已完成但实际顺序偏离请求；按真实场面继续，本回合停止额外换位',rev,
                ('matching_receipt','mandatory_ui_closed','board_permutation_observed','placement_deviated'))
    elif action.kind==Kind.PLAY:
        card=next((c for c in before.hand if c.entity_id==action.source),None)
        placed=next((c for c in after.board if c.entity_id==action.source),None)
        from .magnetic import fusion_target,magnetic
        fused=fusion_target(action,before,after,receipt)
        if fused is not None:
            if action.interaction and action.interaction.placement=='magnetize':
                if fused==action.target:return confirmed('匹配处理内已生成源随从对应的磁力附魔，合体到指定己方机械')
            return Outcome(Status.RESOLVED,f'磁力源已因合体消耗并附加到实体 {fused}；未独立入场，按真实合体结果继续决策',rev,
                ('matching_receipt','source_consumed','mandatory_ui_closed','magnetic_fusion_observed','placement_deviated'))
        if action.interaction and action.interaction.placement=='magnetize':
            if card and placed and receipt.complete and after.interaction is None and action.source not in {c.entity_id for c in after.hand} and placed.controller==before.local_player_id and placed.card_id==card.card_id:
                return Outcome(Status.RESOLVED,'磁力意图未达成，源作为独立随从入场；记录实际落点并重新决策，不重复打出',rev,
                    ('matching_receipt','mandatory_ui_closed','ordinary_summon_observed','placement_deviated'))
            return Outcome(Status.PENDING,'磁力目标合体因果证据尚未确认；源消失或属性上涨不能冒充合体成功',rev)
        if card and card.card_type=="MINION" and placed and placed.controller==before.local_player_id and placed.card_id==card.card_id and action.source not in {c.entity_id for c in after.hand}:
            if action.position is None or placed.position==action.position:
                return confirmed("指定打出随从已进入请求的位置")
            if receipt.complete and after.interaction is None and after.phase!=Phase.CHOICE:
                return Outcome(Status.RESOLVED,'源随从已独立入场，实际位置与请求不同；记录偏差后按真实场面继续',rev,
                    ('matching_receipt','mandatory_ui_closed','ordinary_summon_observed','placement_deviated'))
        if card and card.card_type=='MINION' and receipt.complete and after.interaction is None:
            zones=[(old,new) for eid,old,new in receipt.zone_transitions if eid==action.source]
            entered=next((i for i,event in enumerate(zones) if event==('HAND','PLAY')),None)
            consumed=entered is not None and any(old=='PLAY' and new in ('GRAVEYARD','REMOVEDFROMGAME') for old,new in zones[entered+1:])
            old_controls={c.entity_id for c in before.controls}
            contradictory_fusion=magnetic(card) and any(c.entity_id not in old_controls and
                c.tags.get('MAGNETIC')=='1' and c.tags.get('CREATOR')==str(action.source) for c in after.controls)
            if consumed and not contradictory_fusion and action.source not in {c.entity_id for c in after.hand+after.board}:
                if action.position is not None:
                    return Outcome(Status.RESOLVED,'源随从已入场并在同一完整处理内消耗或死亡；最终位置不可观察',rev,
                        ('matching_receipt','source_consumed','mandatory_ui_closed','placement_unobservable'))
                return confirmed('指定随从已从手牌进入场上，并在同一完整处理内被消耗或死亡；不声称仍留场')
        if card and card.card_type in ('SPELL','BATTLEGROUND_SPELL') and action.source not in {c.entity_id for c in after.hand} and after.interaction is None:
            return Outcome(Status.RESOLVED,'匹配法术处理已开始，原手牌已消耗，强制选择已关闭；具体收益另行核验',rev,
                ('matching_receipt','source_consumed','mandatory_ui_closed'))
    if not receipt.complete:return Outcome(Status.PENDING,"仅观察到动作开始，缺少结束标记且具体后置条件未满足")
    return Outcome(Status.RESOLVED,"匹配的处理已结束；没有足够的专用状态证据证明全部效果",rev,('matching_receipt',))
