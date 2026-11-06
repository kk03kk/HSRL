"""Read-only Win32 discovery/geometry. Deliberately exposes NO input APIs."""
from dataclasses import asdict, dataclass
import ctypes as C
from ctypes import wintypes as W
import hashlib
import json
from pathlib import Path
import sys
import time


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    def point(self,u=.5,v=.5):return self.x+self.width*u,self.y+self.height*v


@dataclass(frozen=True)
class WindowSnapshot:
    hwnd: int
    pid: int
    process_created: int
    executable: str
    title: str
    class_name: str
    client: Box  # Physical screen pixels, NOT logical/DIP pixels.
    monitor: Box
    dpi: int
    monitor_scale_percent: int
    visible: bool
    enabled: bool
    minimized: bool
    cloaked: bool
    foreground: bool
    visible_samples: int
    total_samples: int
    captured_at: float
    dpi_context: str

    @property
    def identity(self):
        return f"{self.hwnd}:{self.pid}:{self.process_created}:{self.executable.lower()}:{self.class_name}"

    @property
    def token(self):
        values=asdict(self);values.pop("captured_at");values.pop("title")
        return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()[:24]

    @property
    def blockers(self):
        reasons=[]
        if not self.visible or self.minimized or self.cloaked:reasons.append("window_not_visible")
        if not self.enabled:reasons.append("window_disabled_or_modal")
        if not self.foreground:reasons.append("window_not_foreground")
        if self.client.width<=0 or self.client.height<=0:reasons.append("empty_client")
        if self.visible_samples!=self.total_samples:reasons.append("occluded_or_offscreen_sample")
        if self.dpi_context!="per_monitor_v2":reasons.append("physical_coordinate_context_unverified")
        return reasons

    def to_dict(self):return {**asdict(self),"identity":self.identity,"token":self.token,"blockers":self.blockers}

    @classmethod
    def from_dict(cls,data):
        from dataclasses import fields
        values={f.name:data[f.name] for f in fields(cls)}
        values["client"]=Box(**values["client"])
        values["monitor"]=Box(**values["monitor"])
        return cls(**values)


class NativeWindows:
    def __init__(self,executable: Path):
        if sys.platform!="win32":raise RuntimeError("Win32 adapter requires Windows")
        self.expected=str(executable.resolve()).casefold()
        self.u=C.WinDLL("user32",use_last_error=True)
        self.k=C.WinDLL("kernel32",use_last_error=True)
        self.d=C.WinDLL("dwmapi",use_last_error=True)
        self.s=C.WinDLL("shcore",use_last_error=True)
        self._bind()
        # These only change THIS process/thread coordinate interpretation.
        self.u.SetProcessDpiAwarenessContext(C.c_void_p(-4))
        self.u.SetThreadDpiAwarenessContext(C.c_void_p(-4))
        self.dpi_context="per_monitor_v2" if self.u.AreDpiAwarenessContextsEqual(self.u.GetThreadDpiAwarenessContext(),C.c_void_p(-4)) else "unknown"

    def _bind(self):
        def bind(d,name,args,result):
            f=getattr(d,name);f.argtypes=args;f.restype=result
        bind(self.u,"SetProcessDpiAwarenessContext",[W.HANDLE],W.BOOL)
        bind(self.u,"SetThreadDpiAwarenessContext",[W.HANDLE],W.HANDLE)
        bind(self.u,"GetThreadDpiAwarenessContext",[],W.HANDLE)
        bind(self.u,"AreDpiAwarenessContextsEqual",[W.HANDLE,W.HANDLE],W.BOOL)
        self.enum_callback=C.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
        bind(self.u,"EnumWindows",[self.enum_callback,W.LPARAM],W.BOOL)
        bind(self.u,"GetWindowThreadProcessId",[W.HWND,C.POINTER(W.DWORD)],W.DWORD)
        bind(self.u,"GetClientRect",[W.HWND,C.POINTER(W.RECT)],W.BOOL)
        bind(self.u,"ClientToScreen",[W.HWND,C.POINTER(W.POINT)],W.BOOL)
        bind(self.u,"GetWindowTextW",[W.HWND,W.LPWSTR,C.c_int],C.c_int)
        bind(self.u,"GetClassNameW",[W.HWND,W.LPWSTR,C.c_int],C.c_int)
        for name in ("IsWindow","IsWindowVisible","IsWindowEnabled","IsIconic"):
            bind(self.u,name,[W.HWND],W.BOOL)
        bind(self.u,"GetForegroundWindow",[],W.HWND)
        bind(self.u,"WindowFromPoint",[W.POINT],W.HWND)
        bind(self.u,"GetAncestor",[W.HWND,W.UINT],W.HWND)
        bind(self.u,"GetDpiForWindow",[W.HWND],W.UINT)
        bind(self.u,"MonitorFromWindow",[W.HWND,W.DWORD],W.HANDLE)
        bind(self.u,"GetMonitorInfoW",[W.HANDLE,C.c_void_p],W.BOOL)
        bind(self.k,"OpenProcess",[W.DWORD,W.BOOL,W.DWORD],W.HANDLE)
        bind(self.k,"CloseHandle",[W.HANDLE],W.BOOL)
        bind(self.k,"QueryFullProcessImageNameW",[W.HANDLE,W.DWORD,W.LPWSTR,C.POINTER(W.DWORD)],W.BOOL)
        bind(self.k,"GetProcessTimes",[W.HANDLE]+[C.POINTER(W.FILETIME)]*4,W.BOOL)
        bind(self.d,"DwmGetWindowAttribute",[W.HWND,W.DWORD,C.c_void_p,W.DWORD],C.c_long)
        bind(self.s,"GetScaleFactorForMonitor",[W.HANDLE,C.POINTER(C.c_int)],C.c_long)

    def process(self,pid):
        handle=self.k.OpenProcess(0x1000,False,pid)
        if not handle:return None
        try:
            size=W.DWORD(32768);buffer=C.create_unicode_buffer(size.value)
            if not self.k.QueryFullProcessImageNameW(handle,0,buffer,C.byref(size)):return None
            times=[W.FILETIME() for _ in range(4)]
            if not self.k.GetProcessTimes(handle,*[C.byref(x) for x in times]):return None
            return buffer.value,(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime
        finally:self.k.CloseHandle(handle)

    def find(self):
        results=[]
        @self.enum_callback
        def visitor(hwnd,_):
            pid=W.DWORD();self.u.GetWindowThreadProcessId(hwnd,C.byref(pid))
            p=self.process(pid.value)
            if p and str(Path(p[0]).resolve()).casefold()==self.expected and self.u.IsWindowVisible(hwnd):
                results.append(self.snapshot(int(hwnd)))
            return True
        if not self.u.EnumWindows(visitor,0):raise C.WinError(C.get_last_error())
        return results

    def unique(self):
        windows=self.find()
        if len(windows)!=1:raise RuntimeError(f"expected one verified window, found {len(windows)}")
        return windows[0]

    def owns_point(self,hwnd,x,y):
        top=self.u.WindowFromPoint(W.POINT(round(x),round(y)))
        return int(self.u.GetAncestor(top,2) or 0)==hwnd

    def snapshot(self,hwnd):
        if getattr(self,'metrics',None):self.metrics.count('window_full_check')
        if not self.u.IsWindow(hwnd):raise RuntimeError("window_destroyed")
        pid=W.DWORD();self.u.GetWindowThreadProcessId(hwnd,C.byref(pid))
        process=self.process(pid.value)
        if not process or str(Path(process[0]).resolve()).casefold()!=self.expected:raise RuntimeError("window_identity_changed")
        rect=W.RECT();point=W.POINT(0,0)
        if not self.u.GetClientRect(hwnd,C.byref(rect)) or not self.u.ClientToScreen(hwnd,C.byref(point)):
            raise C.WinError(C.get_last_error())
        client=Box(point.x,point.y,rect.right-rect.left,rect.bottom-rect.top)
        title=C.create_unicode_buffer(1024);cls=C.create_unicode_buffer(256)
        self.u.GetWindowTextW(hwnd,title,len(title));self.u.GetClassNameW(hwnd,cls,len(cls))
        cloak=W.DWORD(0)
        hr=self.d.DwmGetWindowAttribute(hwnd,14,C.byref(cloak),C.sizeof(cloak))
        class MONITORINFO(C.Structure):
            _fields_=[("cbSize",W.DWORD),("rcMonitor",W.RECT),("rcWork",W.RECT),("dwFlags",W.DWORD)]
        monitor=self.u.MonitorFromWindow(hwnd,2);info=MONITORINFO();info.cbSize=C.sizeof(info)
        if not self.u.GetMonitorInfoW(monitor,C.byref(info)):raise C.WinError(C.get_last_error())
        m=info.rcMonitor;scale=C.c_int(0)
        self.s.GetScaleFactorForMonitor(monitor,C.byref(scale))
        points=[client.point(x,y) for x in (.05,.25,.5,.75,.95) for y in (.08,.5,.92)]
        visible=sum(self.owns_point(hwnd,*p) for p in points)
        return WindowSnapshot(hwnd,pid.value,process[1],process[0],title.value,cls.value,client,
                              Box(m.left,m.top,m.right-m.left,m.bottom-m.top),self.u.GetDpiForWindow(hwnd),scale.value,
                              bool(self.u.IsWindowVisible(hwnd)),bool(self.u.IsWindowEnabled(hwnd)),
                              bool(self.u.IsIconic(hwnd)),bool(cloak.value) if hr==0 else True,
                              int(self.u.GetForegroundWindow() or 0)==hwnd,visible,len(points),time.time(),self.dpi_context)

    def motion_snapshot(self,hwnd,reference):
        """Fresh identity/focus/geometry without a 15-point desktop grid.

        The executor checks actual cursor ownership during motion and performs
        full snapshots before press/release. Process identity is still read.
        """
        if getattr(self,'metrics',None):self.metrics.count('window_motion_check')
        if not self.u.IsWindow(hwnd):raise RuntimeError('window_destroyed')
        pid=W.DWORD();self.u.GetWindowThreadProcessId(hwnd,C.byref(pid))
        process=self.process(pid.value)
        if not process or pid.value!=reference.pid or process[1]!=reference.process_created or str(Path(process[0]).resolve()).casefold()!=self.expected:
            raise RuntimeError('window_identity_changed')
        rect=W.RECT();origin=W.POINT(0,0)
        if not self.u.GetClientRect(hwnd,C.byref(rect)) or not self.u.ClientToScreen(hwnd,C.byref(origin)):raise C.WinError(C.get_last_error())
        client=Box(origin.x,origin.y,rect.right-rect.left,rect.bottom-rect.top)
        if client!=reference.client or self.u.GetDpiForWindow(hwnd)!=reference.dpi:return self.snapshot(hwnd)
        from dataclasses import replace
        return replace(reference,captured_at=time.time(),foreground=int(self.u.GetForegroundWindow() or 0)==hwnd,
            visible=bool(self.u.IsWindowVisible(hwnd)),enabled=bool(self.u.IsWindowEnabled(hwnd)),minimized=bool(self.u.IsIconic(hwnd)))

    def capture_region(self,window,region,output=None):
        return self._capture(window,output,region)

    def capture(self,window,output: Path | None=None):
        return self._capture(window,output,None)

    def _capture(self,window,output,region):
        from PIL import ImageGrab
        start=time.time()
        if window.minimized or window.client.width<=0 or window.client.height<=0:raise RuntimeError("cannot_capture_minimized_client")
        b=window.client
        r=region or Box(0,0,b.width,b.height)
        if min(r.x,r.y)<0 or min(r.width,r.height)<=0 or r.x+r.width>b.width or r.y+r.height>b.height:raise ValueError('capture_region_outside_client')
        image=ImageGrab.grab(bbox=(b.x+r.x,b.y+r.y,b.x+r.x+r.width,b.y+r.y+r.height),all_screens=True).convert("RGB")
        end=time.time();after=self.snapshot(window.hwnd)
        if output is not None:
            output.parent.mkdir(parents=True,exist_ok=True);image.save(output,compress_level=1)
        # Unsaved live frames are not archive identities. The live provider
        # opts out; read-only collectors keep pixel digests for deduplication.
        digest=hashlib.sha256(output.read_bytes() if output is not None else image.tobytes()).hexdigest() if output is not None or not getattr(self,'defer_unsaved_digest',False) else None
        return image,dict(path=str(output.resolve()) if output is not None else None,start=start,end=end,window_token=window.token,
                          geometry_unchanged=after.token==window.token,method="visible desktop client crop; occluded frames blocked",
                          sha256=digest,sha256_kind='png_file' if output is not None else 'rgb_pixels' if digest is not None else 'not_archived',window_after=after.to_dict())
