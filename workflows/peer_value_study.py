"""Frozen peer rank prediction from public state and contextual board/hand tokens."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from workflows import peer_ppo_panel as P
from workflows.peer_exploration import load_peer
from workflows.peer_finetune_holdout import PROTOCOL, INITIAL, INITIAL_SHA

PLAN_PATH = ROOT / "config/research/peer-value-20261119.json"
PLAN = json.loads(PLAN_PATH.read_bytes())
OUT = Path(os.environ.get("HSAI_VALUE_OUT", PLAN["run"]))
SHARDS = 8


class Scorer(P.Scores):
    def __init__(self, net):
        super().__init__(net, "cuda:0")
        self.hook = net.norm.register_forward_hook(self.remember)
        self.blocked = []
        self.turn = 0
        self.force = None
        self.collect = False
        self.last = None

    def remember(self, module, args, output):
        self.hidden = output.detach()

    def vector(self, f, c, value):
        from hsbg_net.net import C_BOARD, C_HAND
        x = self.hidden[0]
        pieces = [x[0]]
        for zone, offset, length in ((1,C_BOARD,7),(8,C_HAND,10)):
            live = torch.as_tensor(c[offset:offset+length] != 0, device=x.device)
            pieces.append((x[zone:zone+length] * live[:, None]).sum(0) / live.sum().clamp_min(1))
        return np.r_[f[:68], value, torch.cat(pieces).float().cpu().numpy()].astype(np.float32)

    @torch.inference_mode()
    def __call__(self, f, c, mask, seat):
        if self.blocked:
            raise ValueError("unexpected sell restriction")
        inputs = [torch.as_tensor(x,device=self.device)[None] for x in (f,c,mask)]
        logits, value = self.net(*inputs)
        v = float(value[0])
        probs = (logits[0].float()/.55).softmax(0).cpu().numpy().astype(np.float64)
        if not np.isfinite(v) or not np.isfinite(probs).all():
            raise ValueError("nonfinite policy/value")
        if self.collect:
            count = self.counts.get(self.turn, 0) + 1
            self.counts[self.turn] = count
            slot = count - 1 if count <= 3 else int(self.data_rng.integers(count))
            if slot < 3:
                self.rows[self.turn,slot] = self.vector(f,c,v)
        if self.capture_next:
            self.last = self.vector(f,c,v)
            self.capture_next = False
        a = int(self.rng.choice(len(mask), p=probs/probs.sum()))
        if self.force is not None:
            a, self.force = self.force, None
        if not mask[a]:
            raise ValueError("illegal choice")
        pi = np.zeros(len(mask),np.float32)
        pi[a] = 1
        return pi,v

    def begin(self, seed, collect=False):
        self.rng = np.random.default_rng([seed,771])
        self.data_rng = np.random.default_rng([seed,355])
        self.collect, self.capture_next = collect, False
        self.counts, self.rows, self.force, self.last = {}, {}, None, None


class Policy(P.Peer):
    def decide(self, game, seat):
        self.evaluator.turn = int(game.turn)
        return super().decide(game,seat)


def field(split):
    from hsbg.az.train.personal import library_digest
    recipe = json.loads((ROOT/"config/research/human-turn-3702-ppo.json").read_bytes())
    proto,pool = P.Q.config(split, allow_tail_variant=split=="hold")
    if (P.Q.engine_info()["source_hash"] != recipe["engine_source_hash"] or
            library_digest(Path("artifacts/libraries/human-turn-3702-20261118")/split) != recipe["library_digests"][split]
            or (split=="hold" and proto.sha256()!=PROTOCOL) or file_identity(INITIAL)["sha256"] != INITIAL_SHA):
        raise ValueError("asset or protocol drift")
    net,_ = load_peer()
    scorer = Scorer(net)
    return pool,Policy(scorer,proto,False),scorer


def natural(i):
    from workflows.peer_distillation import rollout
    for name,split in (("train","train"),("rank_test","hold")):
        start,n = PLAN[name]
        pool,policy,scorer = field(split)
        rows, labels, seeds, turns = [], [], [], []
        for seed in range(start+i,start+n,SHARDS):
            scorer.begin(seed,collect=True)
            rank = rollout(policy,pool,seed)
            for (turn,slot),vec in sorted(scorer.rows.items()):
                if turn >= 1:
                    rows.append(vec)
                    labels.append(rank-1)
                    seeds.append(seed)
                    turns.append(turn)
        with (OUT/"natural"/f"{name}-{i}.npz").open("xb") as stream:
            np.savez_compressed(stream, x=np.stack(rows), y=labels, seed=seeds, turn=turns)
        scorer.hook.remove()


def parallel(stage, worker_file):
    folder = OUT / stage
    folder.mkdir(parents=True,exist_ok=False)
    atomic_json(folder/"plan.json",dict(spec=file_identity(PLAN_PATH),commit=source_identity(ROOT),
        peer=P.PEER_SHA,initial=INITIAL_SHA,hold_protocol=PROTOCOL))
    def launch(i):
        with (folder/f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable,str(worker_file),str(i)],check=True,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(SHARDS) as pool:
        list(pool.map(launch,range(SHARDS)))
    atomic_json(folder/"complete.json",dict(shards=SHARDS))


if __name__=="__main__":
    torch.set_num_threads(1)
    parallel("natural",__file__) if len(sys.argv)==1 else natural(int(sys.argv[1]))
