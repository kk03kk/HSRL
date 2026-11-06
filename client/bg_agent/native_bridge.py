"""Fixture-bound ObservationProvider, independent UI readiness and planner."""
from dataclasses import dataclass,asdict,replace
import ctypes as C
from ctypes import wintypes as W
import hashlib,json,time
from urllib.request import urlopen,Request
from .model import Observation,Kind
from .windows import NativeWindows

@dataclass(frozen=True)
class FixtureBinding:
    hwnd:int
    pid:int
    process_created:int
    executable:str
    nonce:str
    url:str

class FixtureReadiness:
    def __init__(self,windows,quiet=.06):
        self.windows=windows;self.quiet=quiet;self.signature=None;self.since=0;self.wait_started=None;self.wait_ms=[]
        self.u=windows.u;self.g=C.WinDLL('gdi32',use_last_error=True)
        self.u.GetDC.argtypes=[W.HWND];self.u.GetDC.restype=W.HDC
        self.u.ReleaseDC.argtypes=[W.HWND,W.HDC];self.u.ReleaseDC.restype=C.c_int
        self.g.GetPixel.argtypes=[W.HDC,C.c_int,C.c_int];self.g.GetPixel.restype=W.DWORD
    def inspect(self,packet,window):
        start=time.monotonic();reasons=list(window.blockers)
        if time.time()-packet['published_at']>.5:reasons.append('fixture_snapshot_stale')
        if not packet['ready']:reasons.append('ui_animation')
        if (packet['width'],packet['height'])!=(window.client.width,window.client.height):reasons.append('layout_not_rendered')
        patch=packet['ready_patch'];x=window.client.x+patch['x']+patch['width']/2;y=window.client.y+patch['y']+patch['height']/2
        dc=self.u.GetDC(0)
        try:pixel=int(self.g.GetPixel(dc,round(x),round(y)))
        finally:self.u.ReleaseDC(0,dc)
        rgb=(pixel&255,(pixel>>8)&255,(pixel>>16)&255)
        green=pixel!=0xffffffff and rgb[1]>rgb[0]*1.4 and rgb[1]>rgb[2]*1.2
        if not green:reasons.append('visible_ready_indicator_not_green')
        signature=(window.token,packet['layout_version'],packet['mode'],packet['ready'],green)
        if signature!=self.signature or reasons:self.signature=signature;self.since=start
        if not reasons and start-self.since<self.quiet:reasons.append('ui_quiet_period')
        if reasons and self.wait_started is None:self.wait_started=start
        if not reasons and self.wait_started is not None:self.wait_ms.append((start-self.wait_started)*1000);self.wait_started=None
        return dict(ready=not reasons,reasons=reasons,pixel_rgb=rgb,checked_at=time.time())

class NativeObservationProvider:
    def __init__(self,binding,windows,audit):
        self.binding,self.windows,self.audit=binding,windows,audit
        self.readiness=FixtureReadiness(windows);self.last_packet=None;self.last_window=None;self.last_readiness=None;self.last_signature=None
        from .lifecycle import LifecycleController
        self.lifecycle=LifecycleController(audit)
    def packet(self):
        with urlopen(self.binding.url+'/state',timeout=1) as response:p=json.load(response)
        if p.get('nonce')!=self.binding.nonce or p.get('pid')!=self.binding.pid or p.get('hwnd')!=self.binding.hwnd:
            raise RuntimeError('fixture_session_identity_changed')
        return p
    def control(self,command):
        data=json.dumps(dict(nonce=self.binding.nonce,command=command)).encode()
        with urlopen(Request(self.binding.url+'/control',data=data,headers={'Content-Type':'application/json'}),timeout=1) as response:response.read()
    def observe(self):
        p=self.packet();w=self.windows.snapshot(self.binding.hwnd)
        if w.pid!=self.binding.pid or w.process_created!=self.binding.process_created or w.title!='BG Native Fixture '+self.binding.nonce:
            raise RuntimeError('native_fixture_binding_changed')
        ready=self.readiness.inspect(p,w);obs=Observation.from_dict(p['observation'])
        self.lifecycle.observe(obs,scene=obs.scene)
        obs=replace(obs,captured_at=time.time(),ready=obs.ready and ready['ready'],warnings=obs.warnings+tuple(ready['reasons']))
        signature=(obs.revision,obs.ready,tuple(ready['reasons']),w.token,p['receipt_serial'])
        if signature!=self.last_signature:
            self.audit.write('native_observation',observation=obs.to_dict(),ui_readiness=ready,window=w.to_dict(),label=p['label'],receipt=p['last_delivery'])
            self.last_signature=signature
        self.last_packet,self.last_window,self.last_readiness=p,w,ready
        return obs

@dataclass(frozen=True)
class NativePlan:
    revision:int
    action_key:str
    window_token:str
    created_at:float
    expires_at:float
    receipt_serial:int
    source:tuple|None
    destination:tuple|None
    key:int|None
    layout_version:int
    game_id:str=''
    session:str=''
    request_id:int|None=None
    state_token:str=''
    mouse_button:str='left'
    green_signal:bool=False

class NativeInputPlanner:
    def build(self,action,observation,packet,window,ttl=1.5):
        if observation.revision!=packet['observation']['revision'] or action not in observation.actions:raise ValueError('planner_stale_action')
        if not observation.ready:raise ValueError('planner_ui_not_ready')
        regions=packet['regions']
        def point(name):
            b=regions.get(name)
            if b is None:raise ValueError('unknown_native_region:'+name)
            x,y=b['x']+b['width']/2,b['y']+b['height']/2
            if not 0<=x<window.client.width or not 0<=y<window.client.height:raise ValueError('native_plan_outside_client')
            return window.client.x+x,window.client.y+y
        source=destination=None;key=None
        if action.kind==Kind.DISMISS:key=0x0d
        elif action.kind in (Kind.BUY,Kind.PLAY,Kind.SELECT,Kind.CHOOSE,Kind.REORDER,Kind.SELL,Kind.ACTIVATE):source=point('entity:'+str(action.source))
        elif action.kind==Kind.TARGET:source=point('entity:'+str(action.target))
        elif action.kind in (Kind.ENTER,Kind.END,Kind.REFRESH,Kind.UPGRADE,Kind.CONFIRM):source=point('control:'+str(action.kind))
        else:raise ValueError('unsupported_native_gesture:'+str(action.kind))
        if action.kind==Kind.PLAY:destination=point('region:board')
        if action.kind==Kind.SELL:destination=point('region:bob')
        if action.kind==Kind.REORDER:destination=point(f'position:board:{action.position}')
        now=time.time()
        return NativePlan(observation.revision,action.key,window.token,now,now+ttl,packet['receipt_serial'],source,destination,key,packet['layout_version'])
