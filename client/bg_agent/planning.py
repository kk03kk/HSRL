"""Shadow plans describe paths and postconditions but can never send input."""
from dataclasses import asdict,replace
import time

from .model import Kind, Phase
from .geometry import client_to_screen


EXPECTED={
    Kind.BUY:"specific shop entity becomes owned, or separately verified triple result",
    Kind.SELL:"specific board entity leaves own board",
    Kind.PLAY:"hand source leaves hand; board placement/effect and any required target separately verified",
    Kind.UPGRADE:"tavern tier increases",Kind.REFRESH:"shop entity set changes",
    Kind.CHOOSE:"choice closes and selected outcome is observed",Kind.REORDER:"requested board order observed",
    Kind.HERO_POWER:"matching ability receipt plus specific effect if available",
}


# Public aliases retained for existing adapters and replay tools.
from .action_geometry import play_destination_key,reorder_destination_key,route


def validate_plan(plan,obs,window,now=None):
    reasons=[];now=time.time() if now is None else now
    if now>plan["expires_at"]:reasons.append("plan_expired")
    if obs.revision!=plan["revision"] or obs.game_id!=plan["observation"]["game_id"]:reasons.append("observation_changed")
    if window.token!=plan["window"]["token"]:reasons.append("window_layout_focus_or_identity_changed")
    if window.blockers:reasons.extend(window.blockers)
    return reasons


class ShadowPlanner:
    def __init__(self,row,mappings,*,ttl=.75,replay=False,compact=False):
        self.row,self.mappings,self.ttl,self.replay,self.compact=row,mappings,ttl,replay,compact

    def build(self,obs,decision):
        started=time.perf_counter();created=self.row["capture"]["end"]+.01 if self.replay else time.time();a=decision.action
        mapped={m.key:m for m in self.mappings if m.status=="matched" and m.rect}
        reasons=list(self.row["blockers"]);steps=[];confidences=[]
        geometry=route(a,obs,mapped)
        if geometry.error and a.kind not in (Kind.WAIT,Kind.END):reasons.append(geometry.error)
        reasons.extend('unknown_mapping:'+key for key in geometry.required if key not in mapped)
        if obs.phase==Phase.RECRUIT and "scene:recruit" not in mapped:
            reasons.append("visual_recruit_scene_unconfirmed")
        def point(key):
            m=mapped.get(key)
            if m is None:
                reasons.append("unknown_mapping:"+key);return None
            confidences.append(m.confidence);x,y=m.rect.point()
            b=self.row["window"]["client"]
            return dict(region=key,point_client=[x,y],point_screen=client_to_screen((x,y),b))
        def step(kind,key,**fields):
            p=point(key)
            if p:steps.append(dict(kind=kind,**p,**fields))
        if a.kind==Kind.WAIT:reasons.append("policy_wait")
        elif a not in obs.actions:reasons.append("unadvertised_action")
        elif a.kind==Kind.BUY:
            step("drag_start",geometry.source);step("drag_release",geometry.destination)
        elif a.kind==Kind.SELL:
            step("drag_start",geometry.source);step("drag_release",geometry.destination)
        elif a.kind==Kind.PLAY:
            source_card=next((c for c in obs.hand if c.entity_id==a.source),None)
            hand_target=any(c.entity_id==a.target for c in obs.hand)
            direct_spell=bool(source_card and source_card.card_type in ('SPELL','BATTLEGROUND_SPELL') and a.target is not None and a.sub_option_id is None)
            if a.interaction and a.interaction.placement=='magnetize':
                from .magnetic import magnetic_context
                target=next((c for c in obs.board if c.entity_id==a.target),None)
                if not magnetic_context(obs,a):reasons.append('magnetic_context_unverified')
                step('drag_start',f'entity:{a.source}')
                if target:
                    if f'entity:{a.target}' not in mapped:reasons.append(f'unknown_mapping:entity:{a.target}')
                    step('drag_release',geometry.destination)
                else:reasons.append('magnetic_target_unknown')
            elif direct_spell and hand_target:
                step('click',geometry.source)
                steps.append(dict(kind='await_ui_state',state='target_selection',timeout_ms=1500))
                steps.append(dict(kind='deferred_target',entity_id=a.target,requires_new_observation=True))
            elif direct_spell:
                step('drag_start',geometry.source);step('drag_release',geometry.destination)
            else:
                self._play_steps(a,obs,mapped,steps,reasons,step)
        elif a.kind==Kind.REORDER:
            step("drag_start",f"entity:{a.source}")
            if a.evidence=='verified_board_move':
                destination=reorder_destination_key(a,obs)
                if destination:step('drag_release',destination)
                else:reasons.append('invalid_single_drag_reorder')
            else:step("drag_release",f"position:{a.scope or 'board'}:{a.position}")
        elif a.kind in (Kind.CHOOSE,Kind.SELECT):
            if 'scene:choice' not in mapped:reasons.append('visual_choice_scene_unconfirmed')
            step("click",geometry.source)
        elif a.kind==Kind.ACTIVATE:
            if any(c.entity_id==a.source and (c.tags.get('INTERACTABLE_OBJECT')=='1' or c.card_type=='GAME_MODE_BUTTON') for c in obs.board+obs.controls):
                step('click',geometry.source)
            else:
                step('gesture_proposal',f'entity:{a.source}',gesture='unknown')
                reasons.append('activate_control_protocol_unverified')
            self._branch_steps(a,steps,reasons)
            if a.target is not None:
                steps.append(dict(kind='await_ui_state',state='target_selection',timeout_ms=1500))
                steps.append(dict(kind='deferred_target',entity_id=a.target,requires_new_observation=True))
        elif a.kind==Kind.TARGET:
            step('click_target',geometry.source)
            if not obs.interaction or obs.interaction.mode!='target' or obs.interaction.protocol not in ('verified_target','server_targets','fixture'):
                reasons.append('target_selection_mode_not_visually_verified')
        elif a.kind in (Kind.CONFIRM,Kind.CANCEL):
            step('click',geometry.source)
        elif a.kind in (Kind.REFRESH,Kind.UPGRADE,Kind.HERO_POWER,Kind.FREEZE):
            step("click",geometry.source)
            self._branch_steps(a,steps,reasons)
            if a.target is not None:
                steps.append(dict(kind="await_ui_state",state="target_selection",timeout_ms=1500))
                steps.append(dict(kind='deferred_target',entity_id=a.target,requires_new_observation=True))
        else:reasons.append("unsupported_interaction:"+str(a.kind))
        return self._result(obs,decision,started,created,reasons,steps,confidences)

    @staticmethod
    def _branch_steps(a,steps,reasons):
        if a.sub_option_id is None:return
        from .interaction_protocol import branch_entities
        try:options,selected=branch_entities(a)
        except ValueError as exc:reasons.append(str(exc));return
        steps.append(dict(kind='await_ui_state',state='branch_selection',timeout_ms=3000))
        steps.append(dict(kind='deferred_branch',options=options,entity_id=selected,requires_new_observation=True))

    @staticmethod
    def _play_steps(a,obs,mapped,steps,reasons,step):
            from .magnetic import magnetic
            source=next((c for c in obs.hand if c.entity_id==a.source),None)
            if magnetic(source) and obs.board and a.position not in (None,len(obs.board)+1):
                reasons.append('ordinary_magnetic_play_requires_rightmost_gap')
            destination_key=play_destination_key(a,obs,mapped)
            step("drag_start",f"entity:{a.source}");step("drag_release",destination_key)
            destination=mapped.get(destination_key)
            if destination:
                px,py=destination.rect.point()
                # Unspecified ordinary placement permits the client's normal
                # insertion behavior. Untargeted spells need no empty minion
                # slot at all. Explicit/special/targeted stages remain strict.
                strict=a.position is not None or magnetic(source) or a.sub_option_id is not None or a.target is not None
                for card in obs.board if strict else ():
                    region=mapped.get(f'entity:{card.entity_id}')
                    if region and region.rect.x<=px<=region.rect.x+region.rect.width and region.rect.y<=py<=region.rect.y+region.rect.height:
                        reasons.append('drop_region_overlaps_board_entity_requires_placement_adapter')
                if a.interaction and a.interaction.placement!='normal':reasons.append('special_placement_mapping_unverified')
            ShadowPlanner._branch_steps(a,steps,reasons)
            if a.target is not None:
                steps.append(dict(kind="await_ui_state",state="target_selection",timeout_ms=1500))
                steps.append(dict(kind='deferred_target',entity_id=a.target,requires_new_observation=True))
    def _result(self,obs,decision,started,created,reasons,steps,confidences):
        a=decision.action
        # Visual clickability/menu modality is not inferred from image stability.
        if obs.phase not in (Phase.RECRUIT,Phase.CHOICE):reasons.append("unsupported_ui_phase")
        if obs.phase==Phase.RECRUIT and (obs.turn is None or obs.gold is None):reasons.append("logical_recruit_resources_unknown")
        if obs.warnings:reasons.extend("observation_warning:"+w for w in obs.warnings)
        expires=self.row["capture"]["end"]+self.ttl
        if created>expires:reasons.append("capture_expired")
        context=replace(obs,resolutions=tuple(replace(r,post_state=None) for r in obs.resolutions)) if self.compact else obs
        return dict(schema=1,mode="shadow_replay" if self.replay else "shadow_only",can_execute=False,observation=context.to_dict(),revision=obs.revision,
                    receipt_post_states_omitted=self.compact,
                    decision=asdict(decision),action=asdict(a),window=self.row["window"],capture=self.row["capture"],
                    created_at=created,expires_at=expires,status="mapped_shadow" if not reasons else "refused",
                    confidence=min(confidences) if confidences else 0,rejection_reasons=sorted(set(reasons)),steps=steps,
                    preconditions=["same observation revision and game","same window identity, DPI, client rect and focus",
                                   "fresh unobscured frame with the action's readiness signal","all used regions have reviewed appearance matches",
                                   "UI modality and target availability must be independently verified before future execution"],
                    expected_result=EXPECTED.get(a.kind,"requires dedicated effect adapter"),
                    interaction=asdict(a.interaction) if a.interaction else None,
                    placement_semantics=(dict(requested_position=a.position,
                        mode=a.interaction.placement if a.interaction else 'normal',
                        final_combat_order='requires_separate_reorder_if_needed',
                        release_point_validation='unknown',actual_insertion_validation='not_executed',
                        precision_required=a.position is not None or bool(a.interaction and a.interaction.placement!='normal'))
                        if a.kind==Kind.PLAY else None),
                    mapping=[asdict(m) for m in self.mappings],planning_ms=(time.perf_counter()-started)*1000)
