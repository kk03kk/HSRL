"""Semantic legality, independent of coordinates, policy and executor.

Confirmed means current advertised protocol + checked basic preconditions, not
proof of every card effect or UI clickability. No new card rule is invented.
"""
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
import time

from .model import Kind, Phase


class Certainty(StrEnum):
    CONFIRMED="confirmed"
    INFERRED="inferred"
    UNKNOWN="unknown"


@dataclass(frozen=True)
class LegalAction:
    action: object
    certainty: Certainty
    eligible: bool
    reasons: tuple[str, ...]
    preconditions: tuple[str, ...]
    semantic_kind: str


@dataclass(frozen=True)
class LegalResolution:
    revision: int
    actions: tuple[LegalAction, ...]

    def for_policy(self,obs,allow_inferred=False):
        return replace(obs,actions=tuple(a.action for a in self.actions if a.eligible and
                       (a.certainty==Certainty.CONFIRMED or allow_inferred and a.certainty==Certainty.INFERRED)))

    def to_dict(self):return asdict(self)


class LegalActionResolver:
    def __init__(self,max_age=2):self.max_age=max_age

    def resolve(self,obs,*,now=None):
        now=time.time() if now is None else now
        results=[]
        hand={c.entity_id:c for c in obs.hand};board={c.entity_id:c for c in obs.board}
        shop={c.entity_id:c for c in obs.shop};choices={c.entity_id:c for c in obs.choices}
        fixture=obs.source=="sandbox"
        for a in obs.actions:
            reasons=[];conditions=["advertised_by_observation_adapter","same_revision","fresh_observation"]
            confidence=Certainty.CONFIRMED
            if obs.mode!="GT_BATTLEGROUNDS" and not fixture:reasons.append("unsupported_mode")
            if not obs.ready:reasons.append("logical_snapshot_not_ready")
            if not fixture and (obs.captured_at<=0 or now-obs.captured_at>self.max_age or now<obs.captured_at-.05):reasons.append("stale_observation")
            if obs.warnings and not fixture:reasons.extend("observation_warning:"+w for w in obs.warnings)
            if obs.phase==Phase.COMBAT and a.kind!=Kind.WAIT:reasons.append("combat_no_recruit_input")
            if obs.phase==Phase.CHOICE and a.kind not in (Kind.WAIT,Kind.CHOOSE,Kind.SELECT,Kind.CONFIRM,Kind.CANCEL):reasons.append('blocked_by_forced_choice')
            if obs.interaction and obs.interaction.mode=='target' and a.kind not in (Kind.WAIT,Kind.TARGET,Kind.CANCEL):reasons.append('blocked_by_target_interaction')
            if a.kind in (Kind.CHOOSE,Kind.SELECT,Kind.CONFIRM) and obs.interaction and obs.interaction.request_id is not None:
                if a.choice_id!=obs.interaction.request_id:reasons.append('choice_request_changed')
            if a.evidence=="bg_move_adapter":confidence=Certainty.INFERRED
            verified_ui=obs.source=='hearthstone_live' and a.evidence=='verified_client_ui' and a.kind in (Kind.ENTER,Kind.DISMISS) and obs.phase in (Phase.MENU,Phase.GAME_OVER)
            if not fixture and not verified_ui and a.evidence not in ("server_options","server_choices","bg_move_adapter","verified_board_move","verified_insertion","verified_magnetic","verified_hand_target","verified_phase_wait","verified_target_cancel"):
                reasons.append("unverified_action_evidence")
            if a.kind==Kind.BUY:
                conditions+=( ["source_in_shop","hand_capacity_or_proven_triple","server_price_authorization"] )
                if a.source not in shop:reasons.append("buy_source_not_in_shop")
                if len(obs.hand)>=10:
                    source=shop.get(a.source)
                    copies=[c for c in obs.hand+obs.board if source and source.card_id and c.card_id==source.card_id and c.tags.get('PREMIUM','0')=='0']
                    triple=bool(source and source.card_type=='MINION' and source.tags.get('PREMIUM','0')=='0' and len(copies)==2 and a.option_id is not None and a.evidence=='server_options')
                    if not triple:reasons.append("full_hand_requires_triple_or_overflow_adapter")
                # COST is NOT the shop price. Rely on explicit server option.
                if a.option_id is None and not fixture:confidence=Certainty.INFERRED
                if not fixture:
                    if a.resource_cost is None or a.authorization_source is None or obs.gold is None:confidence=Certainty.INFERRED
                    elif a.resource_cost>obs.gold:reasons.append('insufficient_purchase_gold')
                    authorizer=next((c for c in obs.controls if c.entity_id==a.authorization_source),None)
                    if authorizer is None or authorizer.cost!=a.resource_cost:confidence=Certainty.INFERRED
            elif a.kind==Kind.SELL:
                conditions+=["source_on_own_board"]
                if a.source not in board:reasons.append("sell_source_not_on_board")
            elif a.kind==Kind.PLAY:
                conditions+=["source_in_hand","board_capacity_or_explicit_magnetic_placement"]
                card=hand.get(a.source)
                if not card:reasons.append("play_source_not_in_hand")
                if card and card.tags.get('LITERALLY_UNPLAYABLE') in ('1',1):reasons.append('play_source_explicitly_unplayable')
                if a.position is not None and not 1<=a.position<=len(obs.board)+1:reasons.append('play_insertion_position_out_of_range')
                elif card and card.card_type=="MINION" and len(obs.board)>=7:
                    if not a.interaction or a.interaction.placement!="magnetize":reasons.append("board_full")
                    elif not a.target or a.target not in board:reasons.append("magnetic_target_unknown")
                    else:confidence=Certainty.INFERRED;conditions+=["magnetic_mechanism_requires_adapter"]
            elif a.kind==Kind.ACTIVATE:
                conditions+=["server_advertises_activate","source_ability_enabled"]
                if a.source not in board and not any(c.entity_id==a.source for c in obs.controls) and not fixture:
                    reasons.append("activate_control_entity_not_observed")
                if a.source in board and board[a.source].tags.get("INTERACTABLE_OBJECT") not in ("1",1) and not fixture:
                    confidence=Certainty.INFERRED
                control=board.get(a.source) or next((c for c in obs.controls if c.entity_id==a.source),None)
                if not fixture:
                    raw=control.tags.get('INTERACTABLE_OBJECT_COST') if control and a.source in board else control.cost if control else None
                    try:cost=int(raw) if raw is not None else None
                    except (ValueError,TypeError):cost=None
                    conditions+=['known_activate_cost','sufficient_activate_gold']
                    if cost is None or obs.gold is None:confidence=Certainty.INFERRED
                    elif cost>obs.gold:reasons.append('insufficient_activate_gold')
                    if control and control.tags.get('EXHAUSTED') in ('1',1):reasons.append('control_exhausted')
            elif a.kind in (Kind.SELECT,Kind.CHOOSE):
                conditions+=["current_choice_request","candidate_in_choice"]
                candidates=a.selected or ((a.source,) if a.source is not None else ())
                if any(choices[eid].tags.get('BACON_LOCKED_MULLIGAN_HERO') in ('1',1) for eid in candidates if eid in choices):reasons.append('hero_choice_locked')
                if obs.phase!=Phase.CHOICE or not candidates or not set(candidates)<=set(choices) or len(set(candidates))!=len(candidates):reasons.append("not_current_choice_candidate")
                if a.selected and (not obs.interaction or not (0 if a.kind==Kind.SELECT else obs.interaction.min_select)<=len(a.selected)<=obs.interaction.max_select):reasons.append('selection_count_invalid')
                if obs.interaction and obs.interaction.max_select>1 and obs.interaction.protocol not in ("fixture","verified_multi_select"):
                    reasons.append("multi_select_protocol_unknown")
                # Trinket offers are purchases, not free discover rewards.
                priced=[choices[eid] for eid in candidates if eid in choices and choices[eid].card_type=='BATTLEGROUND_TRINKET']
                if priced and not fixture:
                    conditions+=['trinket_price_observed','sufficient_choice_gold','trinket_confirmation_protocol_verified']
                    if any(c.cost is None for c in priced) or obs.gold is None:confidence=Certainty.INFERRED
                    elif sum(c.cost for c in priced)>obs.gold:reasons.append('insufficient_choice_gold')
                    if not obs.interaction or obs.interaction.protocol!='verified_trinket_select':reasons.append('trinket_confirmation_protocol_unknown')
            elif a.kind==Kind.TARGET:
                conditions+=["targeting_mode_confirmed","advertised_target"]
                if not obs.interaction or obs.interaction.mode!="target":reasons.append("target_mode_unknown")
                elif obs.interaction.protocol not in ('server_targets','verified_target','fixture'):reasons.append('target_protocol_unknown')
            elif a.kind==Kind.CONFIRM:
                conditions+=["selection_count_within_bounds"]
                ctx=obs.interaction
                if not ctx or not ctx.min_select<=len(ctx.selected)<=ctx.max_select:reasons.append("selection_count_invalid")
                if ctx and a.selected and a.selected!=ctx.selected:reasons.append('selected_entities_changed')
                if ctx and ctx.protocol not in ('verified_multi_select','verified_target','verified_trinket_select','fixture'):reasons.append('confirm_protocol_unknown')
                if ctx and ctx.protocol=='verified_trinket_select':
                    priced=[choices.get(eid) for eid in ctx.selected]
                    if any(c is None or c.cost is None or c.card_type!='BATTLEGROUND_TRINKET' for c in priced) or obs.gold is None:reasons.append('trinket_price_unknown')
                    elif sum(c.cost for c in priced)>obs.gold:reasons.append('insufficient_choice_gold')
            elif a.kind==Kind.CANCEL:
                if not obs.interaction:reasons.append("no_active_interaction")
                elif not fixture:
                    from .client_protocol import cancellable_target
                    if not cancellable_target(obs,a):reasons.append("interaction_cannot_cancel")
            elif a.kind==Kind.REORDER:
                group=obs.shop if a.scope=="shop" else obs.board
                if sorted(a.order)!=sorted(c.entity_id for c in group):reasons.append("order_is_not_current_group_permutation")
            elif a.kind==Kind.UPGRADE:
                conditions+=["server_allows_upgrade","known_upgrade_cost_if_available"]
                if obs.upgrade_cost is not None and obs.gold is not None and obs.upgrade_cost>obs.gold:reasons.append("insufficient_upgrade_gold")
                if not fixture and (obs.upgrade_cost is None or obs.gold is None):confidence=Certainty.INFERRED
            elif a.kind in (Kind.REFRESH,Kind.HERO_POWER,Kind.FREEZE):
                control=next((c for c in obs.controls if c.entity_id==a.source),None)
                conditions+=['current_control_cost_observed','sufficient_gold','control_not_exhausted']
                if not fixture:
                    if control is None or (a.kind!=Kind.FREEZE and (control.cost is None or obs.gold is None)):confidence=Certainty.INFERRED
                    elif control.cost is not None and obs.gold is not None and control.cost>obs.gold:reasons.append('insufficient_control_gold')
                    if control and control.tags.get('EXHAUSTED')=='1':reasons.append('control_exhausted')
            elif a.kind not in (Kind.REFRESH,Kind.HERO_POWER,Kind.FREEZE,Kind.END,Kind.ENTER,Kind.DISMISS,Kind.WAIT):
                reasons.append("unsupported_action_kind")
            if a.target is not None and a.target not in {**hand,**board,**shop,**choices} and (not obs.hero or obs.hero.entity_id!=a.target) and not fixture:
                reasons.append("target_entity_not_observed")
            if reasons:confidence=Certainty.UNKNOWN
            results.append(LegalAction(a,confidence,not reasons,tuple(reasons),tuple(conditions),"select" if a.kind==Kind.CHOOSE else str(a.kind)))
        return LegalResolution(obs.revision,tuple(results))
