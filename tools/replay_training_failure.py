"""Replay a format-2 collector failure without neural networks or CUDA."""
import argparse
import json
from pathlib import Path
import taverngym as tg


def replay(path, allow_engine_change=False):
    f=json.loads(Path(path).read_text(encoding="utf-8"))
    if f.get("format")!=2:
        raise ValueError("legacy snapshot has no action trace; exact failure replay is unavailable")
    source=f.get("native",{}).get("source_hash")
    if source and source!=tg.build_info()["source_hash"] and not allow_engine_change:
        raise ValueError("engine changed; --allow-engine-change is required for a cross-version diagnostic")
    env=tg.BatchEnv(1,controlled=range(8)); env.auto_reset=False
    env.load(0,json.dumps(f["initial"]))
    result={"steps":0,"rejected":False,"diagnostic":None,"native":tg.build_info()}
    for tick,actions in enumerate(f["actions"]):
        _,dn,info=env.step(actions)
        diag=env.diagnostics()[0]
        result.update(steps=tick+1, rejected=bool(info["rejected"].any()),
                      diagnostic=diag, errors=info["errors"][0])
        if result["rejected"] or any(diag) or any(d and p==0 for d,p in zip(dn,info["placement"])):
            print(json.dumps(result,ensure_ascii=False))
            return 1
    print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("failure")
    p.add_argument("--allow-engine-change",action="store_true")
    a=p.parse_args()
    raise SystemExit(replay(a.failure,a.allow_engine_change))
