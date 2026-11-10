"""Action-specific identity/causality checks. Silence never authorizes retransmission."""
def settled_refresh(decision,old,new,receipt):
    """Independent, narrow acceptance proof when a refresh root lacks BLOCK_END.

    Keep complete=False. Require the exact submitted option, server root, next
    Options boundary and root-bound replacement of the actual old shop. A cost
    change, an input SendOption, or a new shop alone is never an acknowledgement.
    """
    b=decision.binding;r=receipt;s=r.get('submission') or {};post=r.get('post_state') or {}
    cursor=old.get('log_position') or 0;start=r.get('started_at_log_position') or 0
    if (b.kind!='refresh' or b.option_id is None or r.get('complete') or r.get('closed_by')!='next_options'
            or r.get('block_type')!='PLAY' or s.get('option_id')!=b.option_id or s.get('source')!=b.source
            or s.get('owner')!=old.get('local_player_id') or s.get('sub_option_id')!=-1 or s.get('target')!=0
            or not cursor<s.get('log_position',0)<start<r.get('log_position',0)):
        return False
    if (new.get('local_player_id')!=old.get('local_player_id') or new.get('turn')!=old.get('turn')
            or new.get('mode')!='GT_BATTLEGROUNDS' or new.get('phase')!='recruit'
            or new.get('options_pending') or new.get('client_task_list') is not None or not new.get('ready')
            or (new.get('client_submission_position') or 0)!=s['log_position']):return False
    ids=lambda d:{c['entity_id'] for c in d.get('shop',[])}
    removed=ids(old)-ids(post);added=ids(post)-ids(old)
    if not removed or not added or ids(post)!=ids(new):return False
    changes=r.get('entity_changes',[])
    gone={eid for eid,tag,prior,value in changes if tag=='ZONE' and prior=='PLAY' and value in ('REMOVEDFROMGAME','SETASIDE','GRAVEYARD')}
    bob={c.get('controller') for c in post.get('shop',[])}
    arrived={eid for eid,tag,prior,value in changes if tag=='CONTROLLER' and str(value) in {str(p) for p in bob}}
    return removed<=gone and added<=arrived


def settled_play(decision,old,new,receipt):
    if receipt.get('complete') or receipt.get('closed_by')!='next_options':return False
    return departed_play(decision,old,new,receipt)


def departed_play(decision,old,new,receipt):
    """Exact submitted play + same server root + witnessed hand departure.

    Callers must require either a completed root or its next-Options boundary.
    This proves the submitted source/target, not who selected it, nor that every
    downstream battlecry/discovery or visual animation has completed.
    """
    b=decision.binding;r=receipt;s=r.get('submission') or {};post=r.get('post_state') or {}
    cursor=old.get('log_position') or 0;start=r.get('started_at_log_position') or 0
    source=next((c for c in old.get('hand',[]) if c['entity_id']==b.source),None)
    if (b.kind!='play' or b.option_id is None or not source or source.get('controller')!=old.get('local_player_id')
            or source.get('card_type') not in ('MINION','SPELL','BATTLEGROUND_SPELL')
            or r.get('kind')!='play' or r.get('block_type')!='PLAY' or r.get('source')!=b.source
            or r.get('target')!=b.target or r.get('sub_option_id')!=b.sub_option_id
            or s.get('source')!=b.source or s.get('option_id')!=b.option_id or s.get('owner')!=old.get('local_player_id')
            or s.get('target')!=(b.target or 0) or s.get('sub_option_id')!=(-1 if b.sub_option_id is None else b.sub_option_id)
            or not cursor<s.get('log_position',0)<start<r.get('log_position',0)):
        return False
    if (post.get('mode')!='GT_BATTLEGROUNDS' or post.get('local_player_id')!=old.get('local_player_id')
            or post.get('turn')!=old.get('turn') or new.get('local_player_id')!=old.get('local_player_id')
            or b.source in {c['entity_id'] for c in post.get('hand',[])}):return False
    zones={'3':'HAND','1':'PLAY','4':'GRAVEYARD','5':'REMOVEDFROMGAME','6':'SETASIDE'}
    return any(eid==b.source and tag=='ZONE' and zones.get(str(prior),str(prior))=='HAND'
        and zones.get(str(value),str(value)) in ('PLAY','GRAVEYARD','SETASIDE','REMOVEDFROMGAME')
        for eid,tag,prior,value in r.get('entity_changes',[]))


def settled_shop_activation(decision,old,new,r):
    """Exact targeted activation with root-bound destruction and spawned effect."""
    b=decision.binding;s=r.get('submission') or {};post=r.get('post_state') or {}
    if (b.kind!='activate' or b.target is None or r.get('complete') or r.get('closed_by')!='next_options'
            or r.get('kind')!='activate' or r.get('block_type')!='PLAY' or r.get('source')!=b.source
            or r.get('target')!=b.target or r.get('sub_option_id')!=b.sub_option_id
            or s.get('source')!=b.source or s.get('target')!=b.target or s.get('option_id')!=b.option_id
            or s.get('owner')!=old.get('local_player_id') or s.get('sub_option_id')!=-1
            or not (old.get('log_position') or 0)<s.get('log_position',0)<r.get('started_at_log_position',0)<r.get('log_position',0)
            or post.get('mode')!='GT_BATTLEGROUNDS' or post.get('turn')!=old.get('turn')
            or post.get('local_player_id')!=old.get('local_player_id') or new.get('local_player_id')!=old.get('local_player_id')):return False
    source=next((c for z in ('board','controls') for c in old.get(z,[]) if c['entity_id']==b.source),None)
    current=next((c for z in ('board','controls') for c in post.get(z,[]) if c['entity_id']==b.source),None)
    if not source or not current or source.get('controller')!=old.get('local_player_id') or any(source.get(k)!=current.get(k) for k in ('controller','card_id')):return False
    prior={c['entity_id'] for c in old.get('shop',[])};now={c['entity_id'] for c in post.get('shop',[])}
    changes=r.get('entity_changes',[])
    return (b.target in prior and b.target not in now
        and any(e==b.target and t=='ZONE' and a=='PLAY' and v=='GRAVEYARD' for e,t,a,v in changes)
        and any(e!=b.target and t=='ZONE' and a=='SETASIDE' and v=='PLAY' for e,t,a,v in changes))


def settled_buy(decision,old,new,receipt):
    """A shop purchase whose root lacks BLOCK_END but moved exactly that card.

    The 00:06 run bought BG28_845 (entity 9314): gold 10->8, the card entered
    the hand, but the server root was closed only by the next Options, with no
    recorded SendOption. Require the exact root, PLAY->HAND, controller transfer
    to us and not an earlier turn; keep complete=False and never resend.
    """
    b=decision.binding;r=receipt
    if (b.kind!='buy' or r.get('kind')!='buy' or r.get('block_type')!='PLAY'
            or r.get('source')!=b.source or r.get('complete') or r.get('closed_by')!='next_options'):
        return False
    if not (old.get('log_position') or 0)<(r.get('started_at_log_position') or 0)<(r.get('log_position') or 0):
        return False
    post=r.get('post_state') or new
    if (post.get('mode')!='GT_BATTLEGROUNDS' or post.get('turn',0)<old.get('turn')
            or post.get('local_player_id')!=old.get('local_player_id')
            or new.get('local_player_id')!=old.get('local_player_id')
            or new.get('turn',0)<old.get('turn')):return False
    changes=r.get('entity_changes',[])
    zone=any(eid==b.source and tag=='ZONE' and prior=='PLAY' and value=='HAND'
             for eid,tag,prior,value in changes)
    transferred=any(eid==b.source and tag=='CONTROLLER' and str(value)==str(old['local_player_id'])
                    for eid,tag,_,value in changes)
    return (zone and transferred and b.source in {c['entity_id'] for c in new.get('hand',[])}
            and b.source not in {c['entity_id'] for c in new.get('shop',[])})


def settled_sell(decision,old,new,receipt):
    """A real sell whose root was closed by the next Options, without SendOption.

    The 00:56 run sold entity 3460: it left the board (PLAY->SETASIDE) but the
    root had complete=False and submission=null, so the old verifier left the
    intent pending for 120 s and halted. Require the exact root, ownership at
    the old board, the zone departure, the same player/mode and no earlier turn.
    """
    b=decision.binding;r=receipt
    if (b.kind!='sell' or r.get('kind')!='sell' or r.get('block_type')!='PLAY'
            or r.get('source')!=b.source or r.get('complete') or r.get('closed_by')!='next_options'):
        return False
    if not (old.get('log_position') or 0)<(r.get('started_at_log_position') or 0)<(r.get('log_position') or 0):
        return False
    post=r.get('post_state') or new
    if (post.get('mode')!='GT_BATTLEGROUNDS' or post.get('turn',0)<old.get('turn')
            or post.get('local_player_id')!=old.get('local_player_id')
            or new.get('local_player_id')!=old.get('local_player_id')
            or new.get('turn',0)<old.get('turn')):return False
    owned=next((c for c in old.get('board',[]) if c['entity_id']==b.source),None)
    if not owned or owned.get('controller')!=old.get('local_player_id'):return False
    if b.source in {c['entity_id'] for c in new.get('board',[])}:return False
    changes=r.get('entity_changes',[])
    return any(eid==b.source and tag=='ZONE' and prior=='PLAY'
               and value in ('SETASIDE','GRAVEYARD','REMOVEDFROMGAME')
               for eid,tag,prior,value in changes)


def settled_upgrade(decision,old,new,r):
    """Exact submitted upgrade, root-bound tier change and next Options.

    A higher tier alone could be another action. A missing outer BLOCK_END
    remains incomplete; the matching submission and owned tier delta prove it.
    """
    b=decision.binding;s=r.get('submission') or {};post=r.get('post_state') or {}
    source=next((c for c in old.get('controls',[]) if c['entity_id']==b.source),None)
    tier=old.get('tavern_tier');owner=old.get('local_player_id')
    if (b.kind!='upgrade' or not isinstance(tier,int) or not source or source.get('controller')!=owner
            or 'techup' not in source.get('card_id','').lower()
            or r.get('kind')!='upgrade' or r.get('source')!=b.source or r.get('block_type')!='PLAY'
            or r.get('complete') or r.get('closed_by')!='next_options'
            or s.get('source')!=b.source or s.get('option_id')!=b.option_id or s.get('owner')!=owner
            or s.get('target')!=0 or s.get('sub_option_id')!=-1
            or not (old.get('log_position') or 0)<s.get('log_position',0)<r.get('started_at_log_position',0)<r.get('log_position',0)
            or post.get('mode')!='GT_BATTLEGROUNDS' or post.get('local_player_id')!=owner
            or post.get('turn')!=old.get('turn') or post.get('tavern_tier')!=tier+1
            or new.get('local_player_id')!=owner or new.get('turn',0)<old.get('turn')
            or (new.get('tavern_tier') or 0)<tier+1):return False
    owned={c['entity_id'] for c in old.get('controls',[]) if c.get('controller')==owner
           and c.get('card_type') in ('PLAYER','HERO')}
    return any(eid in owned and tag=='PLAYER_TECH_LEVEL' and str(a)==str(tier) and str(v)==str(tier+1)
               for eid,tag,a,v in r.get('entity_changes',[]))


def settled_activation(decision,old,new,receipt):
    """A click-driven activation whose root was closed by the next Options.

    The 13:41/13:55 runs used a hero power and an activated object; the server
    opened the exact PLAY root (source, our SendOption) but never printed the
    outer BLOCK_END, so the old verifier stayed pending for 120 s and halted
    both games. Accept only: exact submission, exact root identity/order, same
    game/player/turn, an Options boundary close, and a root-bound state change
    (zone/controller/tag) as the effect. complete stays False and nothing is
    resent.
    """
    b=decision.binding;r=receipt;s=r.get('submission') or {};post=r.get('post_state') or {}
    if b.kind not in ('hero_power','activate','dark_gift') or r.get('complete') or r.get('closed_by')!='next_options':
        return False
    expected='activate' if b.kind=='dark_gift' else b.kind
    if (r.get('kind')!=expected or r.get('block_type')!='PLAY' or r.get('source')!=b.source
            or r.get('target')!=b.target or r.get('sub_option_id')!=b.sub_option_id
            or not s or s.get('source')!=b.source or s.get('option_id')!=b.option_id
            or s.get('owner')!=old.get('local_player_id')
            or s.get('sub_option_id')!=(-1 if b.sub_option_id is None else b.sub_option_id)
            or s.get('target')!=(b.target or 0)
            or not (old.get('log_position') or 0)<s.get('log_position',0)<r.get('started_at_log_position',0)<r.get('log_position',0)):
        return False
    # HERO_POWER incomplete roots carry no post_state (the parser only snapshots
    # REFRESH/PLAY/ACTIVATE/UPGRADE on an Options boundary), so for a hero power
    # it is optional but must agree when present. Activations keep requiring it.
    if b.kind=='hero_power':
        if post and (post.get('mode')!='GT_BATTLEGROUNDS'
                     or post.get('local_player_id')!=old.get('local_player_id')
                     or post.get('turn')!=old.get('turn')):return False
    elif (not post or post.get('mode')!='GT_BATTLEGROUNDS'
          or post.get('local_player_id')!=old.get('local_player_id')
          or post.get('turn')!=old.get('turn')):return False
    if (new.get('mode')!='GT_BATTLEGROUNDS' or new.get('local_player_id')!=old.get('local_player_id')
            or new.get('turn',0)<old.get('turn')):return False
    changes=r.get('entity_changes',[])
    # The effect must be bound to this root: a zone/controller/tag change, or
    # (hero power) the power becoming exhausted/activated for this turn.
    if any(tag in ('ZONE','CONTROLLER') or (eid==b.source and tag in ('EXHAUSTED','HEROPOWER_ACTIVATIONS_THIS_TURN'))
           for eid,tag,_,_ in changes):
        return True
    power=next((c for c in post.get('controls',[]) if c['entity_id']==b.source),None)
    prior=next((c for c in old.get('controls',[]) if c['entity_id']==b.source),None)
    return bool(power and prior and b.kind=='hero_power'
                and power.get('tags',{}).get('EXHAUSTED')=='1'
                and prior.get('tags',{}).get('EXHAUSTED')!='1')


def verify(decision,before,after,*,require_playback=True,allow_options_boundary=False):
    if before['game_id']!=after['game_id']: return 'context_changed'
    b=decision.binding;old=before['client'];new=after['client']
    if b.kind=='end':
        return 'applied' if new['turn']!=old['turn'] or new['phase'] in ('combat','game_over') else 'pending'
    if require_playback and (new.get('client_task_list') is not None or new.get('options_pending')): return 'pending'
    if after['revision']<=before['revision']: return 'pending'
    if b.branches and b.sub_option_id is None:return 'pending'
    if b.targets and b.target is None:
        # A source/hand transition alone cannot acknowledge a target the model
        # has not selected. Human/default targeting is not our completed action.
        return 'pending'
    expected_kind={'move':'reorder','magnetize':'play','target':'choose','dark_gift':'activate'}.get(b.kind,b.kind)
    receipts=[r for r in new.get('resolutions',[]) if r['kind']==expected_kind and r['source']==b.source
        and (r.get('complete') or allow_options_boundary and
             (settled_refresh(decision,old,new,r) or settled_play(decision,old,new,r)
              or settled_shop_activation(decision,old,new,r) or settled_buy(decision,old,new,r)
              or settled_sell(decision,old,new,r) or settled_upgrade(decision,old,new,r)
              or settled_activation(decision,old,new,r)))
        and r.get('log_position',-1)>(old.get('log_position') or 0)
        and (r.get('started_at_log_position') or r.get('log_position',-1))>(old.get('log_position') or 0)]
    if b.target is not None:
        receipts=[r for r in receipts if r.get('target')==b.target]
    if b.sub_option_id is not None:
        receipts=[r for r in receipts if r.get('sub_option_id')==b.sub_option_id]
    if b.kind=='choose':
        receipts=[r for r in receipts if r.get('block_type')=='CHOICE:'+str(b.choice_id)
                  and r.get('choice_owner') in (None,old.get('local_player_id'))]
    if not receipts: return 'pending'
    def ids(data,zone): return [c['entity_id'] for c in data.get(zone,[])]
    if b.kind=='buy':
        if b.source in ids(new,'hand')+ids(new,'board') and b.source not in ids(new,'shop'): return 'applied'
        # Immediate triple: follow the matching root's explicit copy graph only.
        prior=set(ids(old,'hand')+ids(old,'board'))
        for r in receipts:
            transferred=any(eid==b.source and tag=='CONTROLLER' and str(value)==str(old['local_player_id'])
                for eid,tag,_,value in r.get('entity_changes',[]))
            edges={int(eid):int(value) for eid,tag,_,value in r.get('entity_changes',[])
                   if tag=='COPIED_FROM_ENTITY_ID' and str(value).isdigit()}
            if transferred:
                for eid in set(ids(new,'hand')+ids(new,'board'))-prior:
                    seen=set()
                    while eid in edges and eid not in seen:
                        seen.add(eid);eid=edges[eid]
                        if eid==b.source: return 'applied'
    elif b.kind=='sell':
        if b.source not in ids(new,'board'): return 'applied'
    elif b.kind in ('play','magnetize'):
        if b.source not in ids(new,'hand'):
            if b.kind=='play' and b.position is not None:
                # Unity uses 1-based insertion positions, with zero also
                # denoting the left/default edge. A matched root alone must
                # not turn a witnessed wrong-slot submission into success.
                positions=[r['submission'].get('position') for r in receipts if r.get('submission')
                    and r['submission'].get('source')==b.source
                    and r['submission'].get('owner')==old.get('local_player_id')
                    and r['submission'].get('log_position',0)>(old.get('log_position') or 0)]
                allowed={b.position+1}|({0} if b.position==0 else set())
                if positions and all(p not in allowed for p in positions):return 'rejected'
            if b.kind=='magnetize':
                if b.source in ids(new,'board'):return 'rejected'
                if b.target not in ids(new,'board'):return 'pending'
                attached=any(tag=='ATTACHED' and str(value)==str(b.target)
                    for r in receipts for eid,tag,_,value in r.get('entity_changes',[]))
                if not attached:return 'pending'
            return 'applied'
    elif b.kind=='upgrade':
        if new['tavern_tier']>old['tavern_tier']: return 'applied'
    elif b.kind=='refresh':
        if ids(new,'shop')!=ids(old,'shop'): return 'applied'
    elif b.kind=='freeze':
        # Real 255790 acknowledges each LockAll toggle with a PLAY root on the
        # button entity and increments that button's TAG_SCRIPT_DATA_NUM_1.
        # There is also a TRIGGER root, but it belongs to the 3-of-a-kind
        # checker, not to this button, so it can never prove our click.  The
        # receipt's post_state is the exact end-of-root snapshot; the live shop
        # can remain visually FROZEN for several seconds after that snapshot.
        if any(r.get('source')==b.source and r.get('block_type')=='PLAY'
               and any(eid==b.source and tag=='TAG_SCRIPT_DATA_NUM_1'
                       and str(prior).isdigit() and str(value).isdigit()
                       and int(value)>int(prior)
                       for eid,tag,prior,value in r.get('entity_changes',[]))
               and (r.get('complete') or r.get('closed_by')=='next_options')
               for r in receipts):
            return 'applied'
        frozen=lambda d:tuple(c['tags'].get('FROZEN','0') for c in d.get('shop',[]))
        if frozen(new)!=frozen(old): return 'applied'
        control=lambda d:next((c.get('tags',{}).get('TAG_SCRIPT_DATA_NUM_1') for c in d.get('controls',[]) if c['entity_id']==b.source),None)
        if control(new) is not None and control(old) is not None and control(new)!=control(old):return 'applied'
    elif b.kind=='move':
        # A source that is no longer on the board cannot be reordered; keep the
        # intent unresolved instead of raising out of the receipt poll.
        wanted=ids(old,'board')
        if b.source not in wanted:return 'pending'
        wanted.insert(b.position,wanted.pop(wanted.index(b.source)))
        if ids(new,'board')==wanted: return 'applied'
        # The final board order can legitimately differ from the requested slot
        # when another reorder is interleaved or Unity clamps the insertion.
        # A completed reorder root for *this exact source* after our intent is
        # still a server acknowledgement that the drag was received.  Do not
        # call it "rejected" and then retry the same gesture; the 03:02 game
        # did exactly that and submitted a duplicate reorder.  Any root that
        # reports a different source remains unattributed.
        if any(r.get('source')==b.source
               and (r.get('log_position') or 0)>(old.get('log_position') or 0)
               for r in receipts):
            return 'applied'
        if set(ids(new,'board'))==set(wanted): return 'rejected'
    elif b.kind in ('choose','target','cancel'):
        if (new.get('interaction') or {}).get('request_id')!=b.choice_id: return 'applied'
    elif b.kind in ('hero_power','activate','dark_gift'):
        # Matching completed server root proves acceptance, not effect benefit.
        return 'applied'
    return 'pending'
