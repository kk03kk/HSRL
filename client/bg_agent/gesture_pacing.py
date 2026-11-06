"""Preserve gesture minimum delays without Windows tick-sized oversleep.

Only a private, unnamed timer is used. No global timer resolution changes,
busy spin, game state inference, or OS input. Failures fall back to Event.wait.
"""
import ctypes as C
from ctypes import wintypes as W
import math,sys,time


class WindowsWaitTimer:
    def __init__(self):
        k=self.kernel=C.WinDLL('kernel32',use_last_error=True)
        k.CreateWaitableTimerExW.argtypes=[C.c_void_p,W.LPCWSTR,W.DWORD,W.DWORD]
        k.CreateWaitableTimerExW.restype=W.HANDLE
        k.SetWaitableTimer.argtypes=[W.HANDLE,C.POINTER(C.c_longlong),W.LONG,C.c_void_p,C.c_void_p,W.BOOL]
        k.SetWaitableTimer.restype=W.BOOL
        k.WaitForSingleObject.argtypes=[W.HANDLE,W.DWORD];k.WaitForSingleObject.restype=W.DWORD
        k.CloseHandle.argtypes=[W.HANDLE];k.CloseHandle.restype=W.BOOL
        # HIGH_RESOLUTION; TIMER_MODIFY_STATE | SYNCHRONIZE. Not inherited.
        self.handle=k.CreateWaitableTimerExW(None,None,2,0x100002)
        if not self.handle:raise C.WinError(C.get_last_error())

    def wait(self,seconds):
        due=C.c_longlong(-max(1,math.ceil(seconds*10_000_000)))
        if not self.kernel.SetWaitableTimer(self.handle,C.byref(due),0,None,None,False):
            raise C.WinError(C.get_last_error())
        # A timer/API failure must not strand the input thread indefinitely.
        result=self.kernel.WaitForSingleObject(self.handle,max(100,math.ceil(seconds*1000)+100))
        if result!=0:raise OSError('gesture_timer_wait_failed:'+str(result))

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle);self.handle=None


class GesturePacer:
    def __init__(self,cancel,*,enabled=True,timer_factory=None):
        self.cancel=cancel;self.timer=None;self.attempted=False;self.closed=False
        self.factory=timer_factory or (WindowsWaitTimer if enabled and sys.platform=='win32' else None)
        self.failure=None

    @property
    def mode(self):return 'high_resolution_timer' if self.timer is not None else 'event_wait'

    def wait(self,seconds):
        if not math.isfinite(seconds) or seconds<0:raise ValueError('invalid_gesture_delay')
        if self.cancel.is_set():return True
        if seconds==0:return False
        if not self.attempted and not self.closed:
            self.attempted=True
            if self.factory:
                try:self.timer=self.factory()
                except (OSError,AttributeError) as exc:self.failure=str(exc)
        if self.timer is None:return self.cancel.wait(seconds)
        deadline=time.monotonic()+seconds
        while not self.cancel.is_set():
            remaining=deadline-time.monotonic()
            if remaining<=0:return False
            try:self.timer.wait(min(remaining,.01))
            except OSError as exc:
                self.failure=str(exc);self.close()
                return self.cancel.wait(max(0,deadline-time.monotonic()))
        return True

    def close(self):
        self.closed=True
        if self.timer is not None:
            self.timer.close();self.timer=None
