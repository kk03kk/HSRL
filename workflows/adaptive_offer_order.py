"""Measure preference sensitivity to a shuffled menu while keeping each card, cost, and mask together."""
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.env.obs_std import SLICE_CARD_CHOICES,SLICE_CHOICES,ITEM_F
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from workflows.adaptive_decision_study import POLICY,RUNS,lobby,read,weights

BANK = RUNS/"nus-adaptive-openings-20261120-1"
OUT = RUNS/"nus-adaptive-offer-order-20261120-1"


def measure(evaluator,features,cards,mask,menu,seat):
    n = len(menu)
    if not 2 <= n <= 5:
        raise ValueError("unsupported offer length")
    offsets = np.arange(n)
    base_float = features[SLICE_CHOICES].reshape(-1,ITEM_F)[:n]
    base_cards = cards[SLICE_CARD_CHOICES][:n]
    if not np.array_equal(base_cards, np.asarray(menu)+1):
        raise ValueError("offer cards and physical observation slots disagree")
    predicted = []
    probabilities = []
    for order in permutations(range(n)):
        order = np.asarray(order)
        f,c,m = features.copy(),cards.copy(),mask.copy()
        f[SLICE_CHOICES].reshape(-1,ITEM_F)[:n] = base_float[order]
        c[SLICE_CARD_CHOICES][:n] = base_cards[order]
        m[30+offsets] = mask[30+order]
        prior,_ = evaluator(f,c,m,seat)
        actual = np.zeros(n)
        actual[order] = prior[30:30+n]
        if actual[~mask[30:30+n]].any() or not np.isclose(actual.sum(),1,atol=1e-5):
            raise ValueError("remapped preference assigned mass outside the true offer")
        predicted.append(int(actual.argmax()))
        probabilities.append(actual)
    probs = np.asarray(probabilities)
    mean = probs.mean(0)
    return dict(original=predicted[0],preferred=int(mean.argmax()),by_order=predicted,
                original_p=probs[0].tolist(),mean_p=mean.tolist(),largest_spread=float(np.ptp(probs,axis=0).max()))


class Found(Exception):
    pass


def check(i):
    policy = ACTORS["intent"]
    opening = read(BANK/f"opening-{i}.json")
    result = {}
    def focus(game,action,trace):
        if game.turn!=9 or not 30<=action<46:
            return
        player = json.loads(game.player_json(i%8))
        pending = player["pending"][-1].get("Trinket")
        if not pending or pending["free"] or pending["replace"] is not None:
            return
        f,c,mask = recorder.inputs
        if mask.sum()<2:
            return
        result.update(measure(policy.evaluate,f,c,mask,pending["options"],i%8))
        if result["original"]+30!=action:
            raise ValueError("identity permutation failed to recover the actual chosen card")
        result.update(cards=pending["options"],prefix=trace.hexdigest(),turn=game.turn,seed=opening["seed"],
                      seat=i%8,mask=mask[30:30+len(pending["options"])].tolist())
        raise Found()
    class Recorder:
        model = policy.model
        def __call__(self,f,c,mask,seat):
            self.inputs = (f,c,mask)
            return policy(f,c,mask,seat)
    recorder = Recorder()
    policy.parent.evaluator = recorder
    try:
        lobby.play(ACTORS,i,True,opening["seed"]-i,opening=opening,on_focus=focus)
    except Found:
        pass
    finally:
        policy.parent.evaluator = policy
    atomic_json(OUT/f"sample-{i}.json",result)
    return result


if __name__=="__main__":
    source=source_identity(ROOT)
    bank=read(BANK/"plan.json")
    if source["tracked_dirty"] or bank["policy"]!=POLICY:
        raise ValueError("wrong release or policy")
    ACTORS=weights()
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT/"plan.json",dict(source=source,bank=file_identity(BANK/"plan.json"),policy=POLICY,
        engine_lock=bank["engine_lock"],live_pin=bank["live_pin"],job=os.environ["SLURM_JOB_ID"],
        method="All permutations with card/features/mask moved together. Use first natural T9 ordinary paid trinket with two or more legal options."))
    with ProcessPoolExecutor(16,mp_context=mp.get_context("fork")) as pool:
        raw=list(pool.map(check,range(bank["n"])))
    rows=[r for r in raw if r]
    unstable=sum(len(set(r["by_order"]))>1 for r in rows)
    clear=sum(r["original"]!=r["preferred"] and
              r["mean_p"][r["preferred"]]-r["mean_p"][r["original"]]>=.05 for r in rows)
    atomic_json(OUT/"complete.json",dict(games=len(raw),menus=len(rows),unstable=unstable,
        mean_changes=sum(r["original"]!=r["preferred"] for r in rows),clear_changes=clear,
        max_spread=max((r["largest_spread"] for r in rows),default=0),eligible=unstable>=12 and clear>=8))
