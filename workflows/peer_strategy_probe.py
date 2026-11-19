"""Test temporary, opportunity-matched peer strategy conditioning on natural games."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from workflows import peer_ppo_panel as P
from workflows.peer_exploration import load_peer
from hsbg.runtime.artifacts import atomic_json
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import compare, estimate

OUT = Path("artifacts/runs/nus-peer-strategy-20261119-1")
SEED = 241130000
GAMES = 256


def choose(owned, flags):
    cards = np.unique(owned[owned > 0])
    if len(cards) < 2:
        return None
    hits = flags[:, cards]
    frequency = hits.sum(0)
    weight = np.log(46.0 / (1.0 + frequency))
    scores = hits @ weight
    order = np.argsort(-scores, kind="stable")
    best, second = order[:2]
    # Demand two distinct pieces and at least one reasonably specific one.
    if (hits[best].sum() < 2 or not np.any(hits[best] & (frequency <= 8))
            or scores[best] - scores[second] < 1.0):
        return None
    return int(best)


class Conditional(P.Scores):
    def __init__(self, actor, device, config):
        super().__init__(actor, device)
        self.vocab = config["z_vocab"]
        self.flags = actor.zcard[:-1].detach().ne(0).any(-1).cpu().numpy()
        self.selected = None
        self.checked = False

    @torch.inference_mode()
    def __call__(self, features, cards, mask, seat):
        if self.blocked:
            raise RuntimeError("unexpected sell restriction")
        if self.now == 8 and not self.checked:
            from hsbg_net.net import C_BOARD, C_HAND
            self.selected = choose(np.r_[cards[C_BOARD:C_BOARD+7], cards[C_HAND:C_HAND+10]], self.flags)
            self.checked = True
        active = self.selected is not None and 8 <= self.now <= 11
        if not active:
            return super().__call__(features, cards, mask, seat)
        inputs = [torch.as_tensor(v, device=self.device)[None] for v in (features, cards, mask)]
        z = torch.tensor([self.selected], device=self.device)
        logits, value = self.net(*inputs, z=z)
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            raise ValueError("invalid conditioned output")
        a = int(logits[0].argmax())
        if not mask[a]:
            raise ValueError("illegal conditional choice")
        pi = np.zeros(245, np.float32)
        pi[a] = 1
        return pi, float(value[0])


class Strategy(P.Peer):
    def reset(self):
        super().reset()
        self.evaluator.checked = False
        self.evaluator.selected = None

    def decide(self, game, seat):
        self.evaluator.now = int(game.turn)
        return super().decide(game, seat)


def shard(index):
    torch.set_num_threads(1)
    proto, pool = P.Q.config("train")
    actor, ck = load_peer()
    plain = P.Peer(P.Scores(actor, "cuda:0"), proto, False)
    scores = Conditional(actor, "cuda:0", ck["net_config"])
    strategy = Strategy(scores, proto, False)
    records, options = [], []
    for seed in range(SEED + index, SEED + GAMES, 8):
        parent = P.trial(plain, pool, seed, event_turns=True)
        student = P.trial(strategy, pool, seed, event_turns=True)
        if scores.selected is None and parent != student:
            raise ValueError("unconditioned control changed")
        records.append(dict(parent=parent, candidate=student))
        if scores.selected is not None:
            options.append(dict(seed=seed, option=scores.vocab[scores.selected],
                gain=student["metrics"]["utility"] - parent["metrics"]["utility"]))
    atomic_json(OUT / f"panel-{index}.json", records)
    atomic_json(OUT / f"options-{index}.json", options)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUT / "plan.json", dict(source=source_identity(ROOT), origin=P.PEER_SHA,
        split="train", seed=SEED, games=GAMES, choose_turn=8, default_again=12,
        gate="Require >=20% coverage; paired utility lower CI >0 on eligible games; common T12 log lower CI >=-0.05. Only then independently test holdout."))
    def launch(i):
        with (OUT / f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable, __file__, str(i)], check=True, stdout=log, stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(launch, range(8)))
    report = compare(OUT, SEED, GAMES)
    options = [v for i in range(8) for v in json.loads((OUT / f"options-{i}.json").read_bytes())]
    report["options"] = {name: sum(v["option"] == name for v in options) for name in sorted({v["option"] for v in options})}
    report["eligible_utility"] = estimate([v["gain"] for v in options])
    report["pass"] = (len(options) >= .20 * GAMES and report["eligible_utility"]["ci95"][0] > 0
                      and report["delta"]["log12"]["ci95"][0] >= -.05)
    atomic_json(OUT / "complete.json", report)


if __name__ == "__main__":
    main() if len(sys.argv) == 1 else shard(int(sys.argv[1]))
