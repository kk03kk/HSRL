"""进程树CPU/RSS与GPU原始采样；MIG物理设备利用率不得冒充本作业利用率。"""
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def proc_stat(text):
    fields=text[text.rfind(')')+2:].split()
    return dict(ppid=int(fields[1]),ticks=int(fields[11])+int(fields[12]),
                rss=int(fields[21])*(os.sysconf('SC_PAGE_SIZE') if hasattr(os,'sysconf') else 4096))


def tree_snapshot(pid):
    states={}
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            states[int(directory.name)]=proc_stat((directory/'stat').read_text())
        except (OSError,ValueError,IndexError):
            pass
    members={pid}
    while True:
        added={p for p,s in states.items() if s['ppid'] in members}-members
        if not added:
            break
        members.update(added)
    return {p:s for p,s in states.items() if p in members}


class ResourceMonitor:
    def __init__(self,pid,path,cpus):
        self.pid,self.path,self.cpus=pid,Path(path),cpus
        self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self.loop,daemon=True)
        self.rows=[]
    def start(self):
        self.thread.start()
    def stop(self):
        self.stop_event.set();self.thread.join(timeout=8)
    def loop(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        previous=None
        with self.path.open('w',encoding='utf-8') as stream:
            while not self.stop_event.is_set():
                now=time.monotonic()
                states=tree_snapshot(self.pid) if Path('/proc').exists() else {}
                ticks=sum(s['ticks'] for s in states.values())
                cores=None
                if previous and states:
                    cores=max(0,ticks-previous[1])/os.sysconf('SC_CLK_TCK')/(now-previous[0])
                row=dict(monotonic=now,pids=sorted(states),cpu_cores=cores,
                    rss_sum_bytes=sum(s['rss'] for s in states.values()))
                previous=(now,ticks)
                # 原样保留N/A；物理卡可能承载其它MIG实例，不能解释为本作业负载。
                for name,query in [('gpu_physical','--query-gpu=uuid,name,utilization.gpu,utilization.memory,memory.used'),
                                   ('gpu_process','--query-compute-apps=pid,gpu_uuid,used_gpu_memory')]:
                    try:
                        r=subprocess.run(['nvidia-smi',query,'--format=csv,noheader,nounits'],
                            capture_output=True,text=True,timeout=3)
                        lines=r.stdout.splitlines()
                        if name=='gpu_process':
                            lines=[line for line in lines if line.split(',')[0].strip() in {str(p) for p in states}]
                        row[name]=lines if r.returncode==0 else ['unavailable']
                    except (OSError,subprocess.TimeoutExpired):
                        row[name]=['unavailable']
                self.rows.append(row)
                stream.write(json.dumps(row)+'\n');stream.flush()
                self.stop_event.wait(5)
    def summary(self):
        cpu=[r['cpu_cores'] for r in self.rows if r['cpu_cores'] is not None]
        return dict(samples=len(self.rows),cpu_mean_cores=sum(cpu)/len(cpu) if cpu else None,
            cpu_max_cores=max(cpu) if cpu else None,allocated_cpus=self.cpus,
            rss_sum_peak_bytes=max((r['rss_sum_bytes'] for r in self.rows),default=0),
            gpu_utilization_attributable_to_job=False,
            note='CPU进程退出采样可能少计；RSS求和含共享页重复。GPU原始物理计数不能归因为MIG实例利用率，N/A不是0。')
