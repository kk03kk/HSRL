"""ctypes SendInput transport. No drivers, hooks, injection or message spoofing."""
import ctypes as C
from ctypes import wintypes as W
import time

ULONG_PTR=C.c_size_t
class MOUSEINPUT(C.Structure):
    _fields_=[('dx',W.LONG),('dy',W.LONG),('mouseData',W.DWORD),('dwFlags',W.DWORD),('time',W.DWORD),('dwExtraInfo',ULONG_PTR)]
class KEYBDINPUT(C.Structure):
    _fields_=[('wVk',W.WORD),('wScan',W.WORD),('dwFlags',W.DWORD),('time',W.DWORD),('dwExtraInfo',ULONG_PTR)]
class HARDWAREINPUT(C.Structure):
    _fields_=[('uMsg',W.DWORD),('wParamL',W.WORD),('wParamH',W.WORD)]
class INPUTUNION(C.Union):
    _fields_=[('mi',MOUSEINPUT),('ki',KEYBDINPUT),('hi',HARDWAREINPUT)]
class INPUT(C.Structure):
    _anonymous_=('data',)
    _fields_=[('type',W.DWORD),('data',INPUTUNION)]

def absolute_point(x,y,desktop):
    left,top,width,height=desktop
    if width<=0 or height<=0 or not left<=x<left+width or not top<=y<top+height:
        raise ValueError('point_outside_virtual_desktop')
    return min(65535,int((x-left+.5)*65536/width)),min(65535,int((y-top+.5)*65536/height))

class WindowsInput:
    TAG=0x48530600
    def __init__(self,audit):
        self.audit=audit;self.u=C.WinDLL('user32',use_last_error=True)
        self.u.SendInput.argtypes=[W.UINT,C.POINTER(INPUT),C.c_int];self.u.SendInput.restype=W.UINT
        self.u.GetAsyncKeyState.argtypes=[C.c_int];self.u.GetAsyncKeyState.restype=C.c_short
        self.u.GetSystemMetrics.argtypes=[C.c_int];self.u.GetSystemMetrics.restype=C.c_int
        self.mouse_down=False;self.right_down=False;self.keys=set();self.events=0
    def pressed(self,vk):return bool(self.u.GetAsyncKeyState(vk)&0x8000)
    def send(self,event,label):
        start=time.perf_counter();C.set_last_error(0)
        sent=int(self.u.SendInput(1,C.byref(event),C.sizeof(INPUT)))
        self.audit.write('send_input',label=label,requested=1,inserted=sent,win_error=C.get_last_error(),
                         point_screen_requested=getattr(self,'requested_point',None),
                         struct_size=C.sizeof(INPUT),extra_info=self.TAG,elapsed_ms=(time.perf_counter()-start)*1000)
        self.events+=sent
        if sent!=1:raise OSError('SendInput_inserted_zero; UIPI cause cannot be inferred from error code')
    def move(self,x,y):
        desktop=tuple(self.u.GetSystemMetrics(i) for i in (76,77,78,79))
        dx,dy=absolute_point(x,y,desktop)
        self.requested_point=(x,y)
        self.send(INPUT(type=0,mi=MOUSEINPUT(dx,dy,0,0x8000|0x4000|0x0001,0,self.TAG)),'move')
    def button(self,down):
        # Track an attempted down before SendInput/audit can raise. Cleanup
        # must release a button even if the OS accepted it but logging failed.
        if down:self.mouse_down=True
        self.send(INPUT(type=0,mi=MOUSEINPUT(0,0,0,2 if down else 4,0,self.TAG)),'left_down' if down else 'left_up')
        if not down:self.mouse_down=False
    def right_button(self,down):
        if down:self.right_down=True
        self.send(INPUT(type=0,mi=MOUSEINPUT(0,0,0,8 if down else 16,0,self.TAG)),'right_down' if down else 'right_up')
        if not down:self.right_down=False
    def key(self,vk,down):
        if vk not in (0x0d,0x1b,0x77):raise ValueError('keyboard_key_not_in_fixture_allowlist')
        if down:self.keys.add(vk)
        self.send(INPUT(type=1,ki=KEYBDINPUT(vk,0,0 if down else 2,0,self.TAG)),f'key_{vk}_{down}')
        if not down:self.keys.discard(vk)
    def release(self,cancel=False,*,right_cancel=False):
        errors=[]
        if cancel and self.mouse_down:
            try:
                if right_cancel:
                    self.right_button(True)
                    try:time.sleep(.03)
                    finally:self.right_button(False)
                else:self.key(0x1b,True);self.key(0x1b,False)
            except OSError as exc:errors.append(str(exc))
        if self.mouse_down:
            try:self.button(False)
            except OSError as exc:errors.append(str(exc))
        if getattr(self,'right_down',False):
            try:self.right_button(False)
            except OSError as exc:errors.append(str(exc))
        for vk in tuple(self.keys):
            try:self.key(vk,False)
            except OSError as exc:errors.append(str(exc))
        self.audit.write('input_cleanup',mouse_down=self.mouse_down,keys=sorted(self.keys),errors=errors)
        if errors:raise OSError('input_release_failed:'+','.join(errors))
