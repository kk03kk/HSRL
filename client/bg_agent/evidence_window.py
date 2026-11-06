"""只保留现成帧的有界内存窗口；不截图、不压缩、不执行I/O。"""
from collections import OrderedDict


class EvidenceWindow:
    def __init__(self,max_bytes=24*1024**2,max_frames=4):
        self.max_bytes=max_bytes;self.max_frames=max_frames
        self.frames=OrderedDict();self.bytes=0;self.pinned=None;self.anchor=None
        self.context=None;self.dropped=0

    def remember(self,key,image,sample):
        context=(sample.get('session'),sample['observation'].get('game_id'),sample['observation'].get('turn'))
        if context!=self.context:
            for old in list(self.frames):
                if old!=self.pinned:self._remove(old)
            self.anchor=None;self.context=context
        raw=getattr(image,'image',image);cost=raw.width*raw.height*3
        if cost>self.max_bytes:self.dropped+=1;return False
        if key in self.frames:self._remove(key)
        self.frames[key]=(image,sample,cost);self.bytes+=cost
        full=getattr(image,'is_full',True)
        if full:self.anchor=key
        while self.bytes>self.max_bytes or len(self.frames)>self.max_frames:
            victim=next((k for k in self.frames if k not in (self.pinned,self.anchor,key)),None)
            if victim is None:victim=next((k for k in self.frames if k not in (self.pinned,key)),key)
            self._remove(victim);self.dropped+=1
        return key in self.frames

    def _remove(self,key):
        entry=self.frames.pop(key,None)
        if entry:self.bytes-=entry[2]
        if self.anchor==key:self.anchor=None

    def pin(self,key):
        self.pinned=key if key in self.frames else None

    def diagnostic(self):
        # 优先保留输入前像素、已捕获的完整场景，再保留最后结果帧。
        keys=list(dict.fromkeys(k for k in (self.pinned,self.anchor,next(reversed(self.frames),None)) if k in self.frames))
        return [(k,*self.frames[k][:2]) for k in keys]


def recording_status(provider,audit):
    """明确证据缺失，不将游戏完成等同于归档完整；不访问输入或截图API。"""
    work=getattr(provider,'archive_queue',None)
    return dict(archive_drained=getattr(provider,'archive_drained',None),
        archive_pending=getattr(work,'unfinished_tasks',None),
        storage_error=getattr(provider,'storage_error',None),
        image_quota_reached=getattr(provider,'evidence_truncated',None),
        images_skipped=getattr(provider,'evidence_skipped',0),
        context_frames_not_queued=getattr(provider,'context_frames_not_queued',0),
        audit_rotations=getattr(audit,'rotations',None),audit_omitted_records=getattr(audit,'omitted_records',None),
        capture_scope='scheduled_evidence_not_every_frame')
