"""Summarize sealed paired records and surface reproducible board combinations."""

import argparse
from collections import Counter
from functools import lru_cache
import gzip
from itertools import combinations
import json
from pathlib import Path
from statistics import mean, median
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.summarize_player150_strategy import MOTIFS, TYPES

MODELS = ("0750", "r1", "3286")
PAIRS = (("0750", "r1"), ("0750", "3286"), ("3286", "r1"))
CARDS = {c["id"]: c for c in json.loads((ROOT / "engine/crates/hsbg-engine/data/cards.json").read_bytes())["cards"]}
MOTIFS = MOTIFS | {
    "felboar_hogrider": ("QUILBOAR", ("BG28_633", "BG31_323")),
}


def read(path):
    return json.loads(path.read_bytes())


@lru_cache(maxsize=None)
def resamples(n):
    return np.random.default_rng(61121).integers(0, n, size=(6000, n))


def interval(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"n": 0, "mean": None, "ci": [None, None]}
    low, high = np.quantile(x[resamples(len(x))].mean(axis=1), [0.025, 0.975])
    return {"n": len(x), "mean": float(x.mean()), "ci": [float(low), float(high)]}


def paired(rows, measure):
    return {"-".join((a, b)): interval([measure(r[a]) - measure(r[b]) for r in rows])
            for a, b in PAIRS}


def at(arm, turn):
    return arm["turns"].get(str(turn))


def log_board(arm, turn):
    return np.log1p(at(arm, turn)["strength"])


def basic(rows):
    side = {}
    for model in MODELS:
        arms = [r[model] for r in rows]
        ranks = [a["rank"] for a in arms]
        side[model] = dict(rank=mean(ranks), top4=mean(x <= 4 for x in ranks), first=mean(x == 1 for x in ranks),
                           reach12=mean(at(a, 12) is not None for a in arms), rank_counts=dict(Counter(ranks)),
                           decisions=mean(a["decisions"] for a in arms), injections=sum(a["injections"] for a in arms),
                           tail=sum(a["tail"] for a in arms))
    common = [r for r in rows if all(at(r[m], 9) and at(r[m], 12) for m in MODELS)]
    return dict(models=side, rank=paired(rows, lambda a: a["rank"]),
                top4=paired(rows, lambda a: int(a["rank"] <= 4)),
                first=paired(rows, lambda a: int(a["rank"] == 1)),
                reach12=paired(rows, lambda a: int(at(a, 12) is not None)),
                common_growth_9_to_12=paired(common, lambda a: log_board(a, 12) - log_board(a, 9)))


def turns(rows):
    out = {}
    for turn in range(1, 1 + max(int(t) for r in rows for m in MODELS for t in r[m]["turns"])):
        result = {}
        for model in MODELS:
            stages = [at(r[model], turn) for r in rows if at(r[model], turn)]
            if not stages:
                continue
            actions = {kind for a in stages for kind in a["counts"]}
            result[model] = dict(n=len(stages), m1=median(a["strength"] for a in stages),
                                 m2=mean(a["tier"] for a in stages), health=mean(a["health"] for a in stages),
                                 gold=mean(a["gold"] for a in stages), actions=mean(a["actions"] for a in stages),
                                 forced=sum(a["forced"] for a in stages),
                                 action_mean={k: mean(a["counts"].get(k, 0) for a in stages) for k in sorted(actions)})
        common = [r for r in rows if all(at(r[m], turn) for m in MODELS)]
        result["common_log_strength"] = paired(common, lambda a: log_board(a, turn))
        result["common_tier"] = paired(common, lambda a: at(a, turn)["tier"])
        out[turn] = result
    return out


def normal(card):
    return CARDS[card].get("normal") or card


def sustained(arm):
    boards = {int(t): {normal(c[0]) for c in s["board"]} for t, s in arm["turns"].items() if int(t) >= 8}
    pairs = {}
    for t, current in boards.items():
        for pair in combinations(sorted(current & boards.get(t + 1, set())), 2):
            pairs.setdefault(pair, t)
    return pairs


def declared(rows, games):
    out = {}
    for tag, (tribe, cards) in MOTIFS.items():
        pair = tuple(sorted(cards))
        eligible = [(r, g) for r, g in zip(rows, games, strict=True) if r["0750"]["lobby"] & (1 << TYPES.index(tribe))]
        counts = {m: sum(pair in g[m] for _, g in eligible) for m in MODELS}
        comparisons = {"-".join((a, b)): interval([int(pair in g[a]) - int(pair in g[b]) for _, g in eligible])
                       for a, b in PAIRS}
        timing = {m: [g[m][pair] for _, g in eligible if pair in g[m]] for m in MODELS}
        out[tag] = dict(tribe=tribe, cards=list(pair), eligible=len(eligible), counts=counts,
                        differences=comparisons, median_start={m: median(t) if t else None for m, t in timing.items()})
    return out


def tendencies(rows):
    out = {}
    for start, end in ((1, 3), (4, 7), (8, 10), (11, 13)):
        common = [r for r in rows if all(at(r[m], t) for m in MODELS for t in range(start, end + 1))]
        if not common:
            continue
        measures = {
            "tier": lambda a, t: at(a, t)["tier"],
            "remaining_gold": lambda a, t: at(a, t)["gold"],
            "actions": lambda a, t: at(a, t)["actions"],
        }
        for verb in ("buy", "sell", "refresh", "upgrade", "play", "magnetize", "move", "freeze", "activate"):
            measures[verb] = lambda a, t, k=verb: at(a, t)["counts"].get(k, 0)
        out[f"{start}-{end}"] = {
            name: {m: mean(fn(r[m], t) for r in common for t in range(start, end + 1)) for m in MODELS}
            for name, fn in measures.items()
        } | {"n": len(common)}
    return out


def inspect_candidate(pair, all_rows):
    out = {}
    cards = set(pair)
    for panel, rows in all_rows.items():
        contexts = []
        scores = {}
        for model in MODELS:
            instances = [(r["seed"], r["arms"][model], sustained(r["arms"][model]).get(pair)) for r in rows]
            seen = [(seed, arm, turn) for seed, arm, turn in instances if turn is not None]
            scores[model] = dict(games=len(seen), rank=mean(a["rank"] for _, a, _ in seen) if seen else None,
                                 first_turn=median(t for _, _, t in seen) if seen else None,
                                 strength_at_formation=median(at(a, t + 1)["strength"] for _, a, t in seen) if seen else None)
            if model == "0750":
                for seed, arm, turn in seen:
                    if len(contexts) >= 3:
                        break
                    contexts.append(dict(seed=seed, hero=at(arm, turn)["hero"], rank=arm["rank"], start=turn,
                                         selected=[e for e in arm["actions"] if e[2] and normal(e[2]) in cards],
                                         boards={t: at(arm, t)["board"] for t in (turn, turn+1)}))
        out[panel] = dict(models=scores, replays=contexts)
    return out


def discovery(by_panel):
    if len(by_panel) < 2:
        return []
    counts = {}
    for panel, rows in by_panel.items():
        counts[panel] = {m: Counter(pair for r in rows for pair in sustained(r["arms"][m])) for m in MODELS}
    first = next(iter(counts.values()))
    signals = []
    for pair in first["0750"]:
        lift = [c["0750"][pair] - max(c[m][pair] for m in ("r1", "3286")) for c in counts.values()]
        if min(lift) >= 2 and sum(c["0750"][pair] for c in counts.values()) >= 8:
            signals.append((sum(lift), sum(c["0750"][pair] for c in counts.values()), pair))
    signals.sort(reverse=True)
    return [dict(cards=pair, names=[CARDS[c].get("name") for c in pair], excess=lift,
                 panels=inspect_candidate(pair, by_panel)) for lift, _, pair in signals[:25]]


def per_panel(records):
    rows = [r["arms"] for r in records]
    games = [{m: sustained(r[m]) for m in MODELS} for r in rows]
    tribe_strength = {}
    for tribe in TYPES:
        eligible = [r for r in rows if r["0750"]["lobby"] & (1 << TYPES.index(tribe))]
        if eligible:
            tribe_strength[tribe] = dict(n=len(eligible), rank=paired(eligible, lambda a: a["rank"]))
    return dict(n=len(rows), results=basic(rows), curves=turns(rows),
                lineups=declared(rows, games), actions_by_phase=tendencies(rows), tribe_strength=tribe_strength)


def read_phase(root, mode, seal, plan):
    name = f"{mode}.jsonl.gz"
    if seal["evidence"][name] != file_identity(root / name):
        raise ValueError(f"record hash mismatch: {name}")
    with gzip.open(root / name, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    grouped = {}
    for panel, spec in plan["panels"].items():
        data = [r for r in rows if r["panel"] == panel]
        expected = [(spec["start"] + i, i % 8) for i in range(spec["triplets"])]
        if [(r["seed"], r["seat"]) for r in data] != expected or any(r["mode"] != mode for r in data):
            raise ValueError(f"wrong paired coordinates: {mode}/{panel}")
        grouped[panel] = data
    if sum(map(len, grouped.values())) != len(rows):
        raise ValueError(f"unrecognized rows: {mode}")
    return grouped


def mask_effect(shared, native):
    if any((a["panel"], a["seed"], a["seat"], a["arms"]["0750"]) !=
           (b["panel"], b["seed"], b["seat"], b["arms"]["0750"])
           for a, b in zip(shared, native, strict=True)):
        raise ValueError("phase pairing or 0750 arm drift")
    return {
        m: dict(rank=interval([b["arms"][m]["rank"] - a["arms"][m]["rank"] for a, b in zip(shared, native, strict=True)]),
                top4=interval([int(b["arms"][m]["rank"] <= 4) - int(a["arms"][m]["rank"] <= 4)
                               for a, b in zip(shared, native, strict=True)]),
                moves=sum(action >= 196 for row in native for _, action, _ in row["arms"][m]["actions"]))
        for m in ("r1", "3286")
    }


def main(root, out):
    seal, plan = read(root / "complete.json"), read(root / "plan.json")
    if seal["evidence"]["plan.json"] != file_identity(root / "plan.json"):
        raise ValueError("plan hash mismatch")
    data = {mode: read_phase(root, mode, seal, plan) for mode in ("shared", "native")}
    result = dict(plan=plan, seal=seal,
                  phases={mode: {panel: per_panel(rows) for panel, rows in panels.items()} for mode, panels in data.items()},
                  move_permission_effect={panel: mask_effect(rows, data["native"][panel])
                                          for panel, rows in data["shared"].items()},
                  discovery_shared=discovery(data["shared"]), discovery_native=discovery(data["native"]))
    atomic_json(out, result)
    return result


def plot(data, path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    phases = ("shared", "native")
    panels = tuple(data["phases"]["shared"])
    names = {"hold": "Held-out EOT", "human": "Human EOT", "bot": "Native bots"}
    masks = {"shared": "Shared 196", "native": "Native action sets"}
    colors = {"0750": "#B43E65", "r1": "#286C9D", "3286": "#24856B"}
    fig, axes = plt.subplots(len(panels) * len(phases), 2, figsize=(10, 3.2 * len(panels) * len(phases)),
                             squeeze=False, layout="constrained")
    for index, (mode, panel) in enumerate((mode, panel) for mode in phases for panel in panels):
        left, right = axes[index]
        pane = data["phases"][mode][panel]
        data_turns = pane["curves"]
        for model in MODELS:
            observations = [(turn, frame[model]) for turn, frame in data_turns.items()
                            if model in frame and frame[model]["n"] >= max(2, pane["n"] / 10)]
            x = [turn for turn, _ in observations]
            left.plot(x, [s["m1"] for _, s in observations], ".-", color=colors[model], label=model)
            right.plot(x, [s["n"] / pane["n"] for _, s in observations], ".-", color=colors[model])
        left.set_yscale("symlog", linthresh=10)
        left.set_ylabel("Board ATK + HP (P50)")
        left.set_title(f"{names[panel]} · {masks[mode]} · n={pane['n']}")
        right.set_ylabel("Reach end of recruit")
        right.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        for ax in (left, right):
            ax.set_xlabel("Turn")
            ax.grid(alpha=0.25)
        if index == 0:
            left.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = main(args.run, args.out)
    if args.plot:
        plot(result, args.plot)
