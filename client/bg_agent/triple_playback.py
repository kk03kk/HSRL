"""三连新实体与其他手牌共存时，等待客户端完成这次日志交接。"""
from .model import Kind


class TriplePlayback:
    def __init__(self):self.pending=set()

    def verified(self,before,after):
        if (before.game_id,before.turn)!=(after.game_id,after.turn):return
        old={c.entity_id for c in before.hand+before.board}
        for c in after.hand:
            if c.entity_id not in old and c.tags.get('PREMIUM')=='1' and any(
                k.startswith('BACON_TRIPLED_BASE_MINION_ID') for k in c.tags):
                self.pending.add((after.game_id,after.turn,c.entity_id))

    def blocks(self,obs,action):
        self.pending={key for key in self.pending if key[:2]==(obs.game_id,obs.turn)}
        key=(obs.game_id,obs.turn,action.source)
        if action.kind!=Kind.PLAY or key not in self.pending:return False
        if obs.client_task_list is None or len(obs.hand)<2:
            self.pending.discard(key);return False
        return True
