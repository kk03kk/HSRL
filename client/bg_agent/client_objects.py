"""独立的客户端对象布局投影；绝不向服务器观测或策略伪造实体。"""
import re

REF=re.compile(r'\[entityName=.*? id=(\d+) zone=(\w+) zonePos=(\d+) cardId=(\w*) player=(\d+)\]')
TYPES={'4':'MINION','5':'SPELL','42':'BATTLEGROUND_SPELL'}
ZONES={'1':'PLAY','3':'HAND','4':'GRAVEYARD','5':'REMOVEDFROMGAME','6':'SETASIDE'}


class ClientObjects:
    def __init__(self):
        self.entities={};self.active=None;self.seen=False;self.position=0;self.version=0

    def snapshot(self):
        copy=ClientObjects();copy.entities={eid:dict(e) for eid,e in self.entities.items()}
        copy.seen=self.seen;copy.position=self.position;copy.version=self.version
        return copy

    def proofs(self,obs):return {zone:self.inspect(obs,zone) for zone in ('hand','shop','board')}

    def feed(self,method,body,position):
        if method=='GameState.DebugPrintPower' and body in ('CREATE_GAME','RESET_GAME'):
            self.entities.clear();self.active=None;self.seen=False;self.version+=1
        if method!='PowerTaskList.DebugPrintPower':return
        self.position=position;self.seen=True;self.version+=1
        if body in ('CREATE_GAME','RESET_GAME'):
            self.entities.clear();self.active=None
            return
        # 引用通常保留旧值，只填空缺，显式TAG_CHANGE才覆盖。
        refs=list(REF.finditer(body))
        for ref in refs:
            eid,zone,pos,cid,controller=ref.groups();entry=self.entities.setdefault(int(eid),{})
            for key,value in dict(ZONE=zone,ZONE_POSITION=pos,card_id=cid,CONTROLLER=controller).items():
                entry.setdefault(key,value)
        created=re.match(r'FULL_ENTITY - Creating ID=(\d+) CardID=(\w*)',body)
        updated=body.startswith(('SHOW_ENTITY','CHANGE_ENTITY','FULL_ENTITY - Updating'))
        if created:
            self.active=int(created[1]);self.entities[self.active]={'card_id':created[2]}
            return
        if updated:
            numeric=re.search(r'Entity=(\d+)\b',body)
            self.active=int(numeric[1]) if numeric else int(refs[0][1]) if refs else None
            if self.active is not None:
                if body.startswith('FULL_ENTITY - Updating'):self.entities[self.active]={}
                cid=re.search(r'CardID=(\w*)',body)
                if cid:self.entities.setdefault(self.active,{})['card_id']=cid[1]
            return
        changed=re.match(r'(?:TAG_CHANGE|HIDE_ENTITY -) Entity=(.*?) tag=(\w+) value=(\S+)',body)
        if changed:
            ref,tag,value=changed.groups();match=REF.search(ref)
            eid=int(match[1]) if match else int(ref) if ref.isdigit() else None
            if eid is not None and tag in ('ZONE','ZONE_POSITION','CONTROLLER','CARDTYPE'):
                self.entities.setdefault(eid,{})[tag]=value
            self.active=None
            return
        tag=re.fullmatch(r'tag=(\w+) value=(\S+)',body)
        if tag and self.active is not None and tag[1] in ('ZONE','ZONE_POSITION','CONTROLLER','CARDTYPE'):
            self.entities[self.active][tag[1]]=tag[2]
        elif not tag:self.active=None

    def inspect(self,obs,zone):
        """只比较布局相关字段；数值动画及不相关行不阻塞当前动作。"""
        desired=tuple(sorted((c.entity_id,c.position,c.card_id) for c in getattr(obs,zone)))
        controllers={c.controller for c in getattr(obs,zone)}
        if zone in ('hand','board'):controllers={obs.local_player_id}
        actual=[]
        for eid,entry in self.entities.items():
            kind=TYPES.get(entry.get('CARDTYPE'),entry.get('CARDTYPE'))
            raw=entry.get('ZONE');render_zone=ZONES.get(raw,raw)
            try:controller=int(entry.get('CONTROLLER',''));pos=int(entry.get('ZONE_POSITION','0'))
            except ValueError:continue
            if kind not in ('MINION','SPELL','BATTLEGROUND_SPELL') or controller not in controllers or pos<=0:continue
            if render_zone!=('HAND' if zone=='hand' else 'PLAY'):continue
            if zone=='board' and kind!='MINION':continue
            actual.append((eid,pos,entry.get('card_id','')))
        actual=tuple(sorted(actual))
        ready=self.seen and actual==desired
        return dict(ready=ready,zone=zone,expected=desired,observed=actual,
            cursor=self.position,reason=None if ready else 'client_object_pending:'+zone)


def bind_capture(before,after):
    """每个区域独立绑定截图前后客户端布局，不用截图之后的对象续命。"""
    if before is None or after is None:return None
    return {zone:{**proof,'capture_consistent':proof['observed']==after[zone]['observed']}
            for zone,proof in before.items()}
