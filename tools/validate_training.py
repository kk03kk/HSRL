"""CPU native batch stress using only advertised legal actions; no training or game client."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import taverngym as tg


def stress(games, out):
    rng=np.random.default_rng(922)
    started=time.monotonic()
    total=deferred=0
    for base in range(0,games,32):
        n=min(32,games-base)
        env=tg.BatchEnv(n,seed=2600201+base//32,controlled=range(8),threads=4,
                        config={"content":"Implemented","max_turns":60,
                                "turn_action_caps":[60],"max_actions_per_turn":60})
        env.auto_reset=False
        done=np.zeros(n*8,dtype=bool)
        initial=[json.loads(env.save(g)) for g in range(n)]
        traces=[]
        for tick in range(12000):
            obs=env.observe(); ready=np.flatnonzero(obs["ready"])
            if not len(ready):
                if not done.all(): raise RuntimeError("stalled before termination")
                break
            actions=env.bot_actions()
            for i in ready:
                if rng.random()<.6: actions[i]=rng.choice(np.flatnonzero(obs["action_mask"][i]))
            traces.append(actions.reshape(n,8).tolist())
            _,dn,info=env.step(actions)
            bad=np.flatnonzero(info["rejected"] | (dn & (info["placement"]==0)))
            diag=env.diagnostics()
            bad_games={int(i)//8 for i in bad}|{g for g,d in enumerate(diag) if d[0] or d[1] or d[2]}
            if bad_games:
                out.parent.mkdir(parents=True,exist_ok=True)
                for g in bad_games:
                    out.with_name(f"stress-failure-{base+g}.json").write_text(json.dumps({
                        "format":2,"native":tg.build_info(),"initial":initial[g],"post":json.loads(env.save(g)),
                        "actions":[row[g] for row in traces],"errors":info["errors"][g],"diagnostic":diag[g]}),encoding="utf-8")
                raise RuntimeError(f"native stress failure in games {bad_games}")
            if np.any(dn & done): raise RuntimeError("duplicate terminal")
            done|=dn
            total+=int(info["applied"].sum())
            deferred+=int(np.count_nonzero(obs["ready"] & ~info["applied"]))
            if done.all(): break
        else: raise RuntimeError("tick guard")
        print(json.dumps({"completed_games":base+n,"decisions":total}),flush=True)
    result={"games":games,"applied_decisions":total,"deferred":deferred,
            "seconds":time.monotonic()-started,"native":tg.build_info()}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games",type=int,default=256)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.games<1: p.error("games must be positive")
    print(json.dumps(stress(args.games,args.out)))
