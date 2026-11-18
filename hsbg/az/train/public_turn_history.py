"""Minimal explicit public history, independent of any teacher or simulator RNG.

The marks mean introduced onto this board by an executed Play action during the
tracked turn. They include Play-created tokens, and exclude unobserved history
before a controller reset. Callers must pass ACTUAL actions, not policy intents.
This module does not change existing observation schemas or policies.
"""
import hashlib,json
import numpy as np

SCHEMA='hsai.public-play-history/v1'
SPEC={'schema':SCHEMA,'slots':7,'mark':'own board entities newly present after an executed minion Play',
      'prune':'remove entities no longer on own board','reset':'new turn or explicit controller reset',
      'action_count':'executed own decisions since the tracked turn/reset, including pending choices',
      'scope':'public own action history; no teacher decisions, hidden state, RNG or opponent information'}
SCHEMA_SHA256=hashlib.sha256(json.dumps(SPEC,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PublicTurnHistory:
    def __init__(self):self.reset()
    def reset(self):self.turn=None;self.entities=set();self.actions=0
    def before(self,view):
        turn=int(view['turn'])
        if self.turn is not None and turn<self.turn:raise ValueError('public history turn moved backwards')
        if self.turn!=turn:self.turn=turn;self.entities=set();self.actions=0
        board=view['me']['board']
        if len(board)>7:raise ValueError('public history board exceeds native width')
        self.entities.intersection_update(m['eid'] for m in board)
        marks=np.zeros(7,dtype=np.uint8)
        for i,m in enumerate(board):marks[i]=int(m['eid'] in self.entities)
        return marks,self.actions
    def after(self,before,after,executed_action):
        action=int(executed_action)
        if self.turn!=int(before['turn']) or not 0<=action<245:
            raise ValueError('public history update has no matching turn or action')
        if 46<=action<126:
            item=before['me']['hand'][(action-46)//8]
            if item['type']=='minion':
                previous={m['eid'] for m in before['me']['board']}
                self.entities.update(m['eid'] for m in after['me']['board'] if m['eid'] not in previous)
        self.entities.intersection_update(m['eid'] for m in after['me']['board'])
        self.actions+=1
    def state_dict(self):
        return {'schema':SCHEMA,'schema_sha256':SCHEMA_SHA256,'turn':self.turn,
                'entities':sorted(self.entities),'actions':self.actions}
    @classmethod
    def restore(cls,state):
        if state.get('schema')!=SCHEMA or state.get('schema_sha256')!=SCHEMA_SHA256:
            raise ValueError('public history schema differs')
        if (state['turn'] is not None and (type(state['turn']) is not int or state['turn']<0)
                or type(state['actions']) is not int or state['actions']<0
                or any(type(e) is not int or e<=0 for e in state['entities'])
                or len(set(state['entities']))!=len(state['entities'])):
            raise ValueError('invalid public history state')
        if state['turn'] is None and (state['entities'] or state['actions']):raise ValueError('uninitialized public history is nonempty')
        obj=cls();obj.turn=state['turn'];obj.actions=state['actions'];obj.entities=set(state['entities']);return obj


def project_own_record(state,seat):
    """Project only the public fields required by the history tracker from a sealed record."""
    player=state['players'][seat]
    return {'turn':state['turn'],'me':{'board':[{'eid':m['eid']} for m in player['board']],
        'hand':[{'type':'minion' if 'Minion' in h['item'] else 'spell'} for h in player['hand']]}}
