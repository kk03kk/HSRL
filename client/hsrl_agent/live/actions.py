"""Server Options/EntityChoices -> identity-bound 245 actions. No action ranking."""
from .contracts import Binding,Blocked

def bind_actions(snapshot):
    data=snapshot['client']
    board,hand,shop=(data.get(k,[]) for k in ('board','hand','shop'))
    groups={z:{c['entity_id']:i for i,c in enumerate(data.get(z,[]))} for z in ('board','hand','shop')}
    entities={c['entity_id']:c for z in ('board','hand','shop','controls') for c in data.get(z,[])}
    out=[]
    if data.get('interaction') or data.get('choices'):
        interaction=data.get('interaction') or {}
        if interaction.get('min_select',1)!=1 or interaction.get('max_select',1)!=1:
            raise Blocked('multi-select is not represented by the trained 245-action policy')
        if not data.get('choices') or len(data['choices'])>16: raise Blocked('choice list missing/overflow')
        if interaction.get('mode')=='branch' and len(interaction.get('branch_options',[]))!=len(data['choices']):
            raise Blocked('branch option count differs from displayed candidates')
        for i,c in enumerate(data['choices']):
            # EntityChoices also includes premium-locked hero slots. Preserve
            # their observation/geometry index, but never authorize their click.
            # Actual 254477 frames show all four heroes CAN_BE_DRAFTED=1 while
            # slots one/four have BACON_LOCKED_MULLIGAN_HERO=1 and a padlock.
            if c.get('card_type')=='HERO':
                tags=c.get('tags',{})
                locked=tags.get('BACON_LOCKED_MULLIGAN_HERO','0')
                draft=tags.get('BACON_HERO_CAN_BE_DRAFTED','1')
                if str(locked) not in ('0','1') or str(draft) not in ('0','1'):
                    raise Blocked('hero choice availability tag is not understood')
                if str(locked)=='1' or str(draft)=='0':continue
            mode=interaction.get('mode')
            branch=interaction.get('branch_options',[])
            if mode=='branch' and branch[i][2]!='NONE':continue
            out.append(Binding(30+i,'target' if mode=='target' else 'branch' if mode=='branch' else 'choose',
                source=c['entity_id'],target=c['entity_id'] if interaction.get('mode')=='target' else None,
                choice_id=interaction.get('request_id'),sub_option_id=branch[i][1] if mode=='branch' else None,
                targets=tuple(branch[i][3]) if mode=='branch' else ()))
        # V1.1 动作规则 forbid_cancel=1：训练/评测在采样前清掉 Cancel(7)，V1.1 从未
        # 见过 cancel 合法。实机若把 7 放进掩码，argmax 可能落在这个未训练的位
        # 上——历史实机回放里出现过 play/cancel 循环。只有当取消是唯一可授权动作
        # 时才保留它，否则空掩码会直接 Blocked 死锁。
        if not out and interaction.get('cancel_authorized'):
            out.append(Binding(7,'cancel',choice_id=interaction.get('request_id')))
        return tuple(out)
    # Native EndTurn means stop recruiting. Real BG has no usable END_TURN
    # server option: realization is waiting, never clicking a fabricated button.
    out.append(Binding(0,'end'))
    powers=snapshot['view'].get('power_entities',[])
    for option in snapshot['options']:
        if option.get('error')!='NONE': continue
        oid=option['id'];eid=option.get('entity');kind=option.get('type')
        if kind in ('END_TURN','ENDTURN'): continue
        c=entities.get(eid)
        if c is None: continue
        cid=c['card_id'].lower()
        targets=tuple(option.get('targets',()))
        branches=tuple((r['entity'],r['id'],r['error'],tuple(r.get('targets',()))) for r in option.get('sub_options',()))
        if branches and (len(branches)!=2 or any(r[0] is None for r in branches)):
            raise Blocked('branch authorization is not represented by two witnessed entities')
        if 'dragbuy' in cid:
            for target in targets:
                if target in groups['shop']:
                    out.append(Binding(8+groups['shop'][target],'buy',source=target,option_id=oid))
        elif 'dragsell' in cid:
            for target in targets:
                if target in groups['board']:
                    out.append(Binding(16+groups['board'][target],'sell',source=target,option_id=oid))
        elif 'techup' in cid: out.append(Binding(3,'upgrade',source=eid,option_id=oid))
        elif 'reroll_button' in cid: out.append(Binding(1,'refresh',source=eid,option_id=oid))
        elif 'lockall' in cid:
            # Freeze is a toggle, not an idempotent "freeze" command.  The
            # engine's trained A_FREEZE means freeze (set_frozen(true)); a
            # second press on an already frozen Tavern unfreezes it.  In the
            # 03:02 game a model Freeze arrived while every shop slot was
            # FROZEN=1; the click cleared the freeze and no new root followed.
            # Do not present that opposite transition under the trained action.
            # The real client exposes no separate unfreeze action in the 245
            # vocabulary, so omit Freeze until at least one slot is unfrozen.
            if not shop or not all(str(c.get('tags',{}).get('FROZEN','0'))=='1' for c in shop):
                out.append(Binding(2,'freeze',source=eid,option_id=oid))
        elif eid in groups['hand']:
            h=groups['hand'][eid]
            if c['card_type']=='MINION':
                if len(board)<7:
                    for pos in range(len(board)+1):
                        out.append(Binding(46+h*8+pos,'play',source=eid,position=pos,option_id=oid,targets=targets,branches=branches))
                if str(c.get('tags',{}).get('MAGNETIC','0'))=='1':
                    for target in targets:
                        if target in groups['board']:
                            b=entities[target]
                            if str(b.get('tags',{}).get('CARDRACE')) in ('17','26','MECHANICAL','ALL'):
                                out.append(Binding(126+h*7+groups['board'][target],'magnetize',source=eid,target=target,option_id=oid))
            elif c['card_type'] in ('SPELL','BATTLEGROUND_SPELL'):
                out.append(Binding(46+h*8,'play',source=eid,option_id=oid,targets=targets,branches=branches))
        elif eid in powers:
            out.append(Binding(4+powers.index(eid),'hero_power',source=eid,option_id=oid,targets=targets,branches=branches))
        elif eid in groups['board']:
            i=groups['board'][eid]
            plain_moves=[o for o in snapshot['options'] if o.get('entity')==eid and o.get('type')=='POWER'
                         and not o.get('has_targets') and not o.get('targets') and not o.get('sub_options')]
            is_move=bool(plain_moves and option is plain_moves[0])
            if c.get('tags',{}).get('INTERACTABLE_OBJECT')=='1' and not is_move:
                out.append(Binding(23+i,'activate',source=eid,option_id=oid,targets=targets))
            if is_move:
                for j in range(len(board)):
                    if i!=j: out.append(Binding(196+i*7+j,'move',source=eid,position=j,option_id=oid))
        elif eid==snapshot['view'].get('dark_gift_entity'):
            out.append(Binding(6,'dark_gift',source=eid,option_id=oid,targets=targets))
    unique={}
    for b in out:
        if not 0<=b.action<245: raise Blocked('action overflow')
        if b.action in unique and unique[b.action]!=b:
            from dataclasses import replace
            # Multiple DragBuy authorizers can have different COST/option IDs
            # while authorizing the same entity/UI gesture. Never merge targets
            # or branches that give one flat action different meanings.
            if replace(unique[b.action],option_id=None)==replace(b,option_id=None): continue
            raise Blocked('ambiguous server action')
        unique[b.action]=b
    return tuple(unique.values())
