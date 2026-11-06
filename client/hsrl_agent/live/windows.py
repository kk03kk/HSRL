"""Win32 I/O is acquired only by explicit user run/calibrate commands."""
import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass,asdict
from pathlib import Path
import sys
import time
from .contracts import Blocked,InputNotSent,FocusUnavailable,digest
from .timing import timing_for,validate_plan

@dataclass(frozen=True)
class Window:
    hwnd:int
    pid:int
    created:int
    exe:str
    rect:tuple
    dpi:int
    def token(self): return digest(asdict(self))

class NativeBackend:
    def __init__(self,executable,stop_file,*,enable_input=False,timing='balanced'):
        if sys.platform!='win32': raise Blocked('Windows required')
        self.executable=str(Path(executable).resolve()).casefold();self.stop_file=Path(stop_file)
        self.enabled=enable_input;self.held=False
        self.timing=timing_for(timing);self.clock=time.monotonic;self.sleep=time.sleep
        self.u=C.WinDLL('user32',use_last_error=True);self.k=C.WinDLL('kernel32',use_last_error=True)
        self.u.SetThreadDpiAwarenessContext.argtypes=[C.c_void_p]
        self.u.SetThreadDpiAwarenessContext.restype=C.c_void_p
        if not self.u.SetThreadDpiAwarenessContext(C.c_void_p(-4 & ((1<<(C.sizeof(C.c_void_p)*8))-1))):
            raise Blocked('per-monitor physical-coordinate context unavailable')
        self.k.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];self.k.OpenProcess.restype=W.HANDLE
        self.k.QueryFullProcessImageNameW.argtypes=[W.HANDLE,W.DWORD,W.LPWSTR,C.POINTER(W.DWORD)]
        self.k.GetProcessTimes.argtypes=[W.HANDLE,*([C.POINTER(W.FILETIME)]*4)]
        self.k.CloseHandle.argtypes=[W.HANDLE]
        self.u.GetForegroundWindow.restype=W.HWND
        self.u.IsIconic.argtypes=[W.HWND];self.u.IsIconic.restype=W.BOOL
        self.u.IsWindowVisible.argtypes=[W.HWND];self.u.IsWindowVisible.restype=W.BOOL
        self.u.GetAsyncKeyState.argtypes=[C.c_int];self.u.GetAsyncKeyState.restype=C.c_short
        self.u.GetSystemMetrics.argtypes=[C.c_int];self.u.GetSystemMetrics.restype=C.c_int
        self.u.GetAncestor.argtypes=[W.HWND,W.UINT];self.u.GetAncestor.restype=W.HWND
        self.u.GetClientRect.argtypes=[W.HWND,C.POINTER(W.RECT)]
        self.u.ClientToScreen.argtypes=[W.HWND,C.POINTER(W.POINT)]
        self.u.GetDpiForWindow.argtypes=[W.HWND];self.u.GetDpiForWindow.restype=W.UINT
        self.u.GetWindowThreadProcessId.argtypes=[W.HWND,C.POINTER(W.DWORD)]
        self.u.WindowFromPoint.argtypes=[W.POINT];self.u.WindowFromPoint.restype=W.HWND
        self.u.SendInput.argtypes=[W.UINT,C.c_void_p,C.c_int];self.u.SendInput.restype=W.UINT
        self.u.GetPhysicalCursorPos.argtypes=[C.POINTER(W.POINT)];self.u.GetPhysicalCursorPos.restype=W.BOOL
        self.enum_callback=C.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
        self.u.EnumWindows.argtypes=[self.enum_callback,W.LPARAM];self.u.EnumWindows.restype=W.BOOL
        self.u.ShowWindow.argtypes=[W.HWND,C.c_int];self.u.ShowWindow.restype=W.BOOL
        self.u.BringWindowToTop.argtypes=[W.HWND];self.u.BringWindowToTop.restype=W.BOOL
        self.u.SetForegroundWindow.argtypes=[W.HWND];self.u.SetForegroundWindow.restype=W.BOOL
    def stopped(self): return self.stop_file.exists() or bool(self.u.GetAsyncKeyState(0x77)&0x8000)
    def _identity(self,hwnd):
        pid=W.DWORD();self.u.GetWindowThreadProcessId(hwnd,C.byref(pid))
        handle=self.k.OpenProcess(0x1000,False,pid.value)
        if not handle: raise Blocked('cannot inspect foreground process')
        try:
            buf=C.create_unicode_buffer(32768);size=W.DWORD(len(buf))
            if not self.k.QueryFullProcessImageNameW(handle,0,buf,C.byref(size)): raise Blocked('process path unavailable')
            times=[W.FILETIME() for _ in range(4)]
            if not self.k.GetProcessTimes(handle,*[C.byref(t) for t in times]): raise Blocked('process creation token unavailable')
            created=(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime
        finally: self.k.CloseHandle(handle)
        return buf.value,pid.value,created
    def activate(self):
        """One startup activation of an existing, uniquely path-verified game.

        Never starts a client, synthesizes Alt keys or continuously steals focus.
        """
        if not self.enabled:raise Blocked('automatic activation is disabled for read-only runs')
        if self.stopped():raise Blocked('STOP/F8 requested')
        found=[]
        @self.enum_callback
        def visit(hwnd,_):
            if not self.u.IsWindowVisible(hwnd) and not self.u.IsIconic(hwnd):return True
            try:identity=self._identity(hwnd)
            except Blocked:return True
            if str(Path(identity[0]).resolve()).casefold()==self.executable:found.append((int(hwnd),identity))
            return True
        if not self.u.EnumWindows(visit,0):raise Blocked('cannot enumerate game windows')
        if len(found)!=1:raise Blocked(f'expected one window for verified Hearthstone executable; found {len(found)}')
        hwnd,identity=found[0]
        if self.u.GetForegroundWindow()!=hwnd or self.u.IsIconic(hwnd):
            self.u.ShowWindow(hwnd,9)  # Restore existing geometry, not maximize.
            for _ in range(3):
                if self.stopped():raise Blocked('STOP/F8 requested')
                if self._identity(hwnd)!=identity:raise Blocked('game process changed during startup activation')
                self.u.BringWindowToTop(hwnd);self.u.SetForegroundWindow(hwnd);self.sleep(.05)
                if self.u.GetForegroundWindow()==hwnd:break
            else:raise Blocked('Windows refused foreground activation of the verified Hearthstone window')
        result=self.window()
        if (result.pid,result.created)!=(identity[1],identity[2]):raise Blocked('activated process identity changed')
        return result
    def window(self):
        if self.stopped(): raise Blocked('STOP/F8 requested')
        hwnd=self.u.GetForegroundWindow()
        if not hwnd or self.u.IsIconic(hwnd) or not self.u.IsWindowVisible(hwnd):
            raise FocusUnavailable('Hearthstone must already be visible and foreground',details=dict(hwnd=int(hwnd or 0)))
        try:executable,pid,created=self._identity(hwnd)
        except Blocked as exc:
            raise FocusUnavailable('foreground process cannot be verified; input paused',
                details=dict(hwnd=int(hwnd),reason=str(exc))) from exc
        if str(Path(executable).resolve()).casefold()!=self.executable:
            raise FocusUnavailable('foreground is not the expected Hearthstone executable',
                details=dict(hwnd=int(hwnd),pid=pid,process_created=created,executable=executable))
        rect=W.RECT();point=W.POINT(0,0)
        if not self.u.GetClientRect(hwnd,C.byref(rect)) or not self.u.ClientToScreen(hwnd,C.byref(point)):
            raise Blocked('client rectangle unavailable')
        return Window(int(hwnd),pid,created,executable,(point.x,point.y,rect.right,rect.bottom),self.u.GetDpiForWindow(hwnd))
    def capture(self):
        from PIL import ImageGrab
        before=self.window();x,y,w,h=before.rect
        image=ImageGrab.grab(bbox=(x,y,x+w,y+h),all_screens=True)
        after=self.window()
        if before.token()!=after.token(): raise Blocked('window changed during capture')
        return image,after
    def _check(self,token):
        if self.window().token()!=token: raise Blocked('window identity/geometry/focus changed')
    def _event(self,flags,point=None,token=None,*,check_position=True):
        if not self.enabled: raise Blocked('input backend is disabled')
        if token is not None:self._check(token)
        class Mouse(C.Structure):
            _fields_=[('dx',C.c_int32),('dy',C.c_int32),('data',C.c_uint32),('flags',C.c_uint32),
                      ('time',C.c_uint32),('extra',C.c_size_t)]
        class Payload(C.Union): _fields_=[('mouse',Mouse)]
        class Input(C.Structure): _fields_=[('type',C.c_uint32),('payload',Payload)]
        expected=40 if C.sizeof(C.c_void_p)==8 else 28
        if C.sizeof(Input)!=expected: raise Blocked('SendInput ABI size mismatch')
        dx=dy=0
        if point is not None:
            sx,sy=point
            vx,vy=self.u.GetSystemMetrics(76),self.u.GetSystemMetrics(77)
            vw,vh=self.u.GetSystemMetrics(78),self.u.GetSystemMetrics(79)
            if not vx<=sx<vx+vw or not vy<=sy<vy+vh: raise Blocked('input outside virtual desktop')
            dx=round((sx-vx)*65535/max(vw-1,1));dy=round((sy-vy)*65535/max(vh-1,1));flags|=0x8001|0x4000
        event=Input(0,Payload(mouse=Mouse(dx,dy,0,flags,0,0x4853524c)))
        C.set_last_error(0)
        inserted=self.u.SendInput(1,C.byref(event),C.sizeof(event))
        trace=getattr(self,'last_input_trace',None)
        if trace is not None:
            trace['events'].append(dict(flags=flags,point=list(point) if point else None,inserted=int(inserted),
                win_error=C.get_last_error(),elapsed_ms=(self.clock()-trace['started'])*1000))
        if inserted!=1: raise Blocked('SendInput not accepted; never retry blindly')
        if point is not None and check_position and hasattr(self.u,'GetPhysicalCursorPos'):
            clock=getattr(self,'clock',time.monotonic);pause=getattr(self,'sleep',time.sleep)
            actual=W.POINT();deadline=clock()+.025
            while True:
                if not self.u.GetPhysicalCursorPos(C.byref(actual)):raise Blocked('physical cursor position unavailable')
                if abs(actual.x-point[0])<=3 and abs(actual.y-point[1])<=3:break
                if clock()>=deadline:raise Blocked(f'cursor coordinate mismatch: requested={point}, actual={(actual.x,actual.y)}')
                pause(.002)
    def _neutral(self,token,rect):
        """Leave the hovered card after an unsent press; never click anything."""
        try:
            x,y,w,h=rect
            point=(x+round(.96*(w-1)),y+round(.91*(h-1)))
            self._event(0,point,token)
            return True
        except Blocked:
            return False
    def execute(self,plan,snapshot):
        self.last_input_trace=dict(started=self.clock(),events=[],completed=False)
        window=token=rect=None
        try:
            validate_plan(plan)
            if any(self.u.GetAsyncKeyState(k)&0x8000 for k in (1,2,0x10,0x11,0x12)):
                raise Blocked('human mouse button/modifier is held')
            window=self.window();token=snapshot['visual']['window_token']
            self._check(token);x,y,w,h=window.rect;rect=window.rect
        except Blocked as exc:raise InputNotSent(str(exc)) from exc
        timing=self.timing;started=self.clock()
        def absolute(p):
            if not all(0<v<1 for v in p): raise Blocked('point outside client')
            pt=(x+round(p[0]*(w-1)),y+round(p[1]*(h-1)))
            owner=self.u.GetAncestor(self.u.WindowFromPoint(W.POINT(*pt)),2)
            if int(owner or 0)!=window.hwnd: raise Blocked('action point occluded')
            return pt
        down_sent=False;hovered=False
        try:
            for row in plan:
                for point in row[1:]:absolute(point)
            for event in plan:
                kind=event[0];start=absolute(event[1])
                self._event(0,start,token);hovered=True
                if kind=='move_pointer':continue
                self.sleep(timing.hover)
                if getattr(self,'pre_press',None):self.pre_press(snapshot)
                if any(self.u.GetAsyncKeyState(k)&0x8000 for k in (1,2,0x10,0x11,0x12)):
                    raise Blocked('human mouse button/modifier changed before press')
                absolute(event[1])
                self._event(0,start,token)  # Verify the intended point after hover.
                if kind=='right_click':
                    down_sent=True
                    self.held='right';self._event(0x0008,token=token);self.sleep(timing.hold)
                    self._event(0x0010,token=token);self.held=False
                else:
                    down_sent=True
                    self.held=True;self._event(0x0002,token=token);self.sleep(timing.hold)
                    if kind=='drag':
                        end=absolute(event[2])
                        drag_start=self.clock()
                        for i in range(1,timing.steps+1):
                            p=(round(start[0]+(end[0]-start[0])*i/timing.steps),round(start[1]+(end[1]-start[1])*i/timing.steps))
                            self._event(0,p,token,check_position=i==timing.steps)
                            self.sleep(max(0,drag_start+timing.drag*i/timing.steps-self.clock()))
                        # Recheck endpoint ownership immediately before release.
                        absolute(event[2])
                    self._event(0x0004,token=token);self.held=False
                    if kind=='double_click':
                        self.sleep(timing.double_gap)
                        if getattr(self,'pre_press',None):self.pre_press(snapshot)
                        absolute(event[1]);self._event(0,start,token)
                        self.held=True;self._event(0x0002,token=token)
                        self.sleep(timing.hold);self._event(0x0004,token=token);self.held=False
                # Let Unity consume button-up while the pointer is still on
                # the control. Moving away in the same frame loses clicks.
                self.sleep(timing.release)
            self.last_input_trace['completed']=True
        except Blocked as exc:
            if not down_sent:
                # A cancelled pre-press check has already moved the real pointer
                # onto the card. Leaving it there mutates hover/tooltips and can
                # hide hero rows or choice rails from every retry. A move-only
                # neutralization is safe; no button is ever pressed here. It is
                # skipped when we never moved (e.g. a human-held button), so a
                # human drag is never pulled to another coordinate.
                if hovered and token is not None and rect is not None:self._neutral(token,rect)
                raise InputNotSent(str(exc)) from exc
            raise
        finally:
            if self.held:
                # Release our own held button even after focus changes. Never
                # click another card/escape as a recovery policy.
                self._event(0x0010 if self.held=='right' else 0x0004);self.held=False
            self.last_timing=dict(profile=timing.name,elapsed_ms=(self.clock()-started)*1000)
