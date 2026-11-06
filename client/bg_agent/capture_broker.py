"""One capture owner with exact-context reuse and genuine client ROI grabs."""
import threading
import time

from .chain_metrics import ChainMetrics
from .windows import Box


class FullCaptureRequired(ValueError):pass


class ClientImage:
    """A real ROI with client-relative crop coordinates, never fake pixels."""
    def __init__(self,image,region,client_size):
        self.image=image;self.region=region;self.size=client_size
        self.is_full=region==Box(0,0,*client_size)
        self._pixels=None

    @property
    def width(self):return self.size[0]
    @property
    def height(self):return self.size[1]
    @property
    def __array_interface__(self):
        if not self.is_full:raise FullCaptureRequired('full_frame_detector_cannot_consume_partial_capture')
        import numpy as np
        if self._pixels is None:self._pixels=np.asarray(self.image)
        return self._pixels.__array_interface__

    def crop(self,bounds):
        x0,y0,x1,y1=map(round,bounds);r=self.region
        if x0<r.x or y0<r.y or x1>r.x+r.width or y1>r.y+r.height:
            raise FullCaptureRequired('requested_pixels_outside_captured_region')
        return self.image.crop((x0-r.x,y0-r.y,x1-r.x,y1-r.y))

    def copy(self):return self.image.copy()
    def convert(self,*a,**kw):return self.image.convert(*a,**kw)
    def save(self,*a,**kw):return self.image.save(*a,**kw)


class CaptureBroker:
    def __init__(self,windows,metrics=None):
        self.windows=windows;self.metrics=metrics or ChainMetrics()
        self.lock=threading.RLock();self.last=None;self.sequence=0

    def invalidate(self):
        with self.lock:self.last=None

    def capture(self,window,output=None,*,stamp=None,region=None,max_age=0):
        requested_at=time.time()
        full=Box(0,0,window.client.width,window.client.height);region=region or full
        if region.width<=0 or region.height<=0 or min(region.x,region.y)<0 or region.x+region.width>full.width or region.y+region.height>full.height:
            raise ValueError('capture_region_outside_client')
        with self.lock:
            now=time.time();old=self.last
            coalesced=bool(old and stamp is not None and old[2]['end']>=requested_at)
            if output is None and (max_age>0 or coalesced) and not window.blockers and old and old[0]==(window.token,stamp):
                image,capture=old[1:];r=image.region if isinstance(image,ClientImage) else full
                if (coalesced or 0<=now-capture['end']<=max_age) and r.x<=region.x and r.y<=region.y and r.x+r.width>=region.x+region.width and r.y+r.height>=region.y+region.height:
                    self.metrics.count('capture_reused')
                    return image,{**capture,'reused_capture':True}
            with self.metrics.measure('capture'):
                if region==full:image,capture=self.windows.capture(window,output)
                else:
                    image,capture=self.windows.capture_region(window,region,output)
                    image=ClientImage(image,region,(full.width,full.height))
            self.sequence+=1
            capture={**capture,'frame_sequence':self.sequence,'reused_capture':False,
                     'region_client':dict(x=region.x,y=region.y,width=region.width,height=region.height)}
            self.metrics.count('capture_full' if region==full else 'capture_roi')
            if capture.get('geometry_unchanged') and not capture.get('window_after',{}).get('blockers'):
                self.last=((window.token,stamp),image,capture)
            else:self.last=None
            return image,capture

    def capture_many(self,window,regions,*,stamp=None,max_age=0):
        regions=list(regions)
        if not regions:return self.capture(window,stamp=stamp,max_age=max_age)
        x=min(r.x for r in regions);y=min(r.y for r in regions)
        right=max(r.x+r.width for r in regions);bottom=max(r.y+r.height for r in regions)
        return self.capture(window,stamp=stamp,region=Box(x,y,right-x,bottom-y),max_age=max_age)
