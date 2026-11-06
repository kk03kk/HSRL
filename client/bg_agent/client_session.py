"""Observed scene transitions and resync boundaries; no network intervention."""
import re
from .connection import IncrementalLog,latest_log_file


class ClientLifecycle:
    def __init__(self):
        self.session=None;self.tail=None;self.pending=False;self.scene=None;self.game=None;self.epoch=0
        from .lifecycle import LifecycleController
        self.controller=LifecycleController()
        from .network_state import NetworkState
        self.network=NetworkState()
    def inspect(self,row):
        session=row['session'];changed=session!=self.session
        if changed:
            self.session=session;self.pending=False;self.scene=None;self.game=None;self.epoch+=1
            newest=latest_log_file(session,'LoadingScreen*.log')
            self.tail=IncrementalLog(newest) if newest else None
        newest=latest_log_file(session,'LoadingScreen*.log')
        if newest and (self.tail is None or self.tail.path!=newest):self.tail=IncrementalLog(newest)
        events=[]
        if self.tail:
            reset,lines=self.tail.poll()
            if reset:self.pending=True;self.epoch+=1
            for line in lines:
                if 'OnScenePreUnload()' in line:self.pending=True;events.append('scene_pre_unload')
                if 'OnSceneLoaded()' in line and (m:=re.search(r'currMode=(\w+)',line)):
                    self.scene=m[1];self.pending=False;events.append('scene_loaded:'+self.scene)
        obs=row['observation'];boundary=self.game is not None and self.game!=obs['game_id']
        if boundary:self.epoch+=1
        self.game=obs['game_id']
        from .model import Observation
        supervised=self.controller.observe(Observation.from_dict(obs),scene=self.scene,transition=self.pending)
        network=self.network.inspect(session)
        return dict(**supervised,scene=self.scene,transition_pending=self.pending,session_changed=changed,
            resync_boundary=boundary,epoch=self.epoch,events=events,
            **network,matching='unknown',
            note='Scene and explicit network events; silence alone does not imply disconnect.')
