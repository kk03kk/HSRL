"""跟踪迟到回执；只有经独立无提交证明重新开放的购买可撤销对应旧票据。"""
class LateResponseWatch:
    def __init__(self):
        self.pending=[]

    def abandon(self,action,before,after):
        self.pending.append((action,before,after.log_position,after.revision))
        self.pending=self.pending[-12:]

    def retire_no_submission(self,action,before):
        """调用方必须已重新证明无提交；不能让新尝试的回执误配为旧尝试迟到。"""
        kept=[];removed=0
        for old,prior,cursor,revision in self.pending:
            if (old==action and prior.game_id==before.game_id and prior.turn==before.turn
                    and prior.log_position==before.log_position and prior.revision==before.revision):removed+=1
            else:kept.append((old,prior,cursor,revision))
        self.pending=kept
        return removed

    def inspect(self,obs,pending=()):
        remaining=[];found=[]
        for action,before,cursor,revision in self.pending:
            if obs.game_id!=before.game_id:continue
            def matches(r):
                newer=(r.log_position>cursor if cursor is not None and r.log_position is not None else r.at_revision>revision)
                return (newer and r.kind==action.kind and r.source==action.source
                    and r.target==action.target and r.sub_option_id==action.sub_option_id
                    and (action.scope is None or r.scope==action.scope)
                    and r.origin in ('server_power','server_choices')
                    and (action.choice_id is None or r.block_type=='CHOICE:'+str(action.choice_id)))
            hit=next((r for r in (*obs.resolutions,*pending) if matches(r)),None)
            if hit is None:remaining.append((action,before,cursor,revision))
            else:found.append(dict(action_key=action.key,source=action.source,receipt_revision=hit.at_revision,
                receipt_log_position=hit.log_position,complete=hit.complete,resend=False))
        self.pending=remaining
        return found
