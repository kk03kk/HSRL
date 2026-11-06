"""Lifecycle supervision shared by native fixtures and real read-only sessions.

AgentController still owns decisions/execution. This class only classifies and
records lifecycle observations; it never invents a queue/connection state.
"""
class LifecycleController:
    def __init__(self,audit=None):
        self.audit=audit;self.state=None;self.game=None;self.terminal=False;self.history=[]
    def observe(self,obs,*,scene=None,transition=False):
        if self.game is not None and obs.game_id!=self.game:self.terminal=False
        self.game=obs.game_id
        if obs.phase=='game_over':self.terminal=True
        if transition:state='transition'
        elif obs.phase=='menu':state='returned_menu' if self.terminal else 'start_menu' if scene=='BACON' or obs.source=='sandbox' else 'menu'
        elif scene=='MATCHMAKING' and obs.source=='sandbox':state='matching'
        elif obs.interaction and obs.interaction.mode=='target':state='target_selection'
        elif obs.phase=='choice':state='hero_selection' if obs.choices and all(c.card_type=='HERO' for c in obs.choices) else 'forced_choice'
        else:state=str(obs.phase)
        if state!=self.state:
            event=dict(previous=self.state,state=state,revision=obs.revision,source=obs.source,game_id=obs.game_id)
            self.history.append(event);self.state=state
            if self.audit:self.audit.write('lifecycle_stage',**event)
        return dict(state=state,terminal_seen=self.terminal,stages_seen=[e['state'] for e in self.history],
                    provenance='fixture explicit scene' if obs.source=='sandbox' else 'real log/modal evidence; unobserved matching remains unknown')
