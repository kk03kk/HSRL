"""真实共享内存反复读取：FD关闭、内容保真、覆盖旧槽不污染返回batch。"""
import os
import uuid
import numpy as np
from hsbg.az.train import process_pool as pp


def test_owned_shm_repeated_reads_close_and_survive_slot_reuse(monkeypatch):
    slots={};name='hsb_owned_'+uuid.uuid4().hex
    arrays={'features':np.arange(64,dtype=np.float32).reshape(8,8),
            'actions':np.arange(8,dtype=np.int64),'masks':np.ones((8,3),dtype=bool),
            'old_logprobs':np.linspace(-3,0,8,dtype=np.float32)}
    meta=pp._shm_put(arrays,slots=slots,slot_idx=0,name=name)
    original=pp._shm_read;handles=[]
    def read(meta):
        a,seg=original(meta);handles.append(seg);return a,seg
    monkeypatch.setattr(pp,'_shm_read',read)
    before=len(os.listdir('/proc/self/fd')) if os.path.isdir('/proc/self/fd') else None
    try:
        for _ in range(700):
            result=pp._shm_read_owned(meta)
            for k,v in arrays.items():
                assert np.array_equal(result[k],v)
                assert result[k].flags.owndata
            assert handles[-1]._buf is None and handles[-1]._mmap is None
        if before is not None:assert len(os.listdir('/proc/self/fd'))<=before+2
        pp._shm_put({k:np.zeros_like(v) for k,v in arrays.items()},slots=slots,slot_idx=0,name=name)
        for k,v in arrays.items():assert np.array_equal(result[k],v)
    finally:pp._shm_close_slot(slots,0)


def test_owned_shm_fp16_upcast_does_not_change_other_fields():
    slots={};name='hsb_owned_'+uuid.uuid4().hex
    arrays={'features':np.array([1.25,2.5],dtype=np.float16),'actions':np.array([2,4],dtype=np.int64)}
    meta=pp._shm_put(arrays,slots=slots,slot_idx=0,name=name)
    try:
        result=pp._shm_read_owned(meta)
        assert result['features'].dtype==np.float32
        assert np.array_equal(result['features'],arrays['features'].astype(np.float32))
        assert result['actions'].dtype==np.int64
    finally:pp._shm_close_slot(slots,0)


def test_shm_put_recovers_from_stale_worker_segment():
    """worker 重启后旧段仍在 /dev/shm：create=True 不能把长训直接打挂。"""
    import uuid as _uuid
    from multiprocessing import shared_memory
    name = 'hsb_owned_' + _uuid.uuid4().hex
    stale = shared_memory.SharedMemory(name=name, create=True, size=pp._SHM_ROUND)
    stale.close()
    slots = {}
    arrays = {'features': np.arange(16, dtype=np.float32).reshape(4, 4),
              'actions': np.arange(4, dtype=np.int64)}
    try:
        meta = pp._shm_put(arrays, slots=slots, slot_idx=0, name=name)
        read = pp._shm_read_owned(meta)
        for k, v in arrays.items():
            assert np.array_equal(read[k], v)
    finally:
        pp._shm_close_slot(slots, 0)
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
