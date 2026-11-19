"""Measure the actual immediate board-stat return of naturally triggered plays."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from workflows.quilboar_study import Recorder,config,policy,play,read,atomic_json,file_identity

JOBS=[Path("artifacts/runs/nus-quilboar-"+k+"-20261119-1") for k in ("choice","investment")]
OUT=Path("artifacts/runs/nus-quilboar-proc-audit-20261119-1")


class ReturnRecorder(Recorder):
    def begin_game(self,game,index,seed,seat):
        super().begin_game(game,index,seed,seat)
        self.measured=[]

    def before(self,game,seat):
        previous=super().before(game,seat)
        if previous is None:
            return None
        p=json.loads(game.player_json(seat))
        size=[1+p["stats"][k] for k in ("gem_atk","gem_hp")]
        return previous,{m["eid"]:m["atk"]+m["hp"] for m in p["board"]},size

    def after(self,game,before,decision,a,forced):
        if before is None:
            return
        prev,old,size=before
        n=len(self.events)
        super().after(game,prev,decision,a,forced)
        if len(self.events)==n:
            return
        p=json.loads(game.player_json(0))
        turn,card,gems=self.events[-1]
        common=[m for m in p["board"] if m["eid"] in old]
        delta=sum(m["atk"]+m["hp"]-old[m["eid"]] for m in common)
        recipients=sum(m["eid"] in prev[2] and m["gems"]>=prev[2][m["eid"]]+2 for m in common)
        self.measured.append(dict(turn=turn,card=card,gems=gems,gem_size=size,
                                  recipients=recipients,immediate_stats=delta))


def shard(i):
    torch.set_num_threads(1)
    proto,pool=config("hold")
    rows=[]
    for root in JOBS:
        sel=read(root/"selection.json")
        if file_identity(sel["path"])!=sel["model"] or read(root/"complete.json")["selected"]!=sel["model"]:
            raise ValueError("different endpoint")
        p=policy(sel["path"],proto)
        for panel in ("natural","special"):
            for result in read(root/panel/f"{i}.json"):
                student=result["student"]
                if not any(9<=e[0]<=12 for e in student["events"]):
                    continue
                rec=ReturnRecorder()
                run=play(p,pool,student["seed"],rec)
                if run!=student:
                    raise ValueError("replay disagrees with blind trajectory")
                rows.append(dict(study=root.name,panel=panel,seed=student["seed"],events=rec.measured))
    atomic_json(OUT/f"{i}.json",rows)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    def worker(i):
        with (OUT/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,__file__,str(i)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(worker,range(8)))
    rows=sum([read(OUT/f"{i}.json") for i in range(8)],[])
    atomic_json(OUT/"complete.json",dict(valid=True,rows=rows,sources=[file_identity(j/"complete.json") for j in JOBS]))


if __name__=="__main__":
    main() if len(sys.argv)==1 else shard(int(sys.argv[1]))
