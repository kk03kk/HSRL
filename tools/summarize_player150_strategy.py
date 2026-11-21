"""Summarize the frozen Player150 strategy assessment without rerunning games."""
import argparse
from collections import Counter
import gzip
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean, median

from summarize_peer_distillation import estimate

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("r1", "final")
TYPES = ("BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECHANICAL", "MURLOC",
         "PIRATE", "QUILBOAR", "UNDEAD", "ABERRATION", "NAGA")
MOTIFS = {
    "azerite_surfacer": ("ELEMENTAL", ("BG28_707", "BG35_881")),
    "quill_rat_rider": ("QUILBOAR", ("BG31_326", "BG31_323")),
    "robot_drone": ("MECHANICAL", ("BG36_366", "BG26_152")),
    "hope_poet": ("DRAGON", ("BG36_364", "BG29_813")),
    "rogers_privateer": ("PIRATE", ("BG33_823", "BG33_825")),
    "felboar_vineweaver": ("QUILBOAR", ("BG28_633", "BG33_883")),
    "eternal_knight_summoner": ("UNDEAD", ("BG25_008", "BG25_009")),
}


def eot(arm, turn):
    return arm["turns"].get(str(turn))


def matched(rows, value):
    return estimate([value(row["final"]) - value(row["r1"]) for row in rows])


def rates(rows, predicate):
    return dict(n=len(rows), **{arm: sum(predicate(row[arm]) for row in rows) for arm in ARMS},
                delta=matched(rows, lambda arm: int(predicate(arm))))


def pair_stats(rows):
    def pane(arm):
        ranks = [row[arm]["rank"] for row in rows]
        return dict(rank=mean(ranks), top4=mean(r <= 4 for r in ranks), first=mean(r == 1 for r in ranks),
                    rank_distribution=dict(Counter(ranks)),
                    decisions=mean(row[arm]["decisions"] for row in rows),
                    injections=sum(row[arm]["injections"] for row in rows), tail=sum(row[arm]["tail"] for row in rows))

    result = {arm: pane(arm) for arm in ARMS}
    result.update(rank_delta=matched(rows, lambda a: a["rank"]),
                  top4_delta=matched(rows, lambda a: int(a["rank"] <= 4)),
                  first_delta=matched(rows, lambda a: int(a["rank"] == 1)),
                  reach12=rates(rows, lambda a: eot(a, 12) is not None))
    common = [r for r in rows if all(eot(r[name], turn) for name in ARMS for turn in (9, 12))]
    result["common_growth_9_12"] = matched(common, lambda a: math.log1p(eot(a, 12)["strength"])
                                             - math.log1p(eot(a, 9)["strength"]))
    rank_changes = [r["final"]["rank"]-r["r1"]["rank"] for r in rows]
    result["better_tied_worse"] = [sum(d < 0 for d in rank_changes), rank_changes.count(0),
                                   sum(d > 0 for d in rank_changes)]
    return result


def turn_stats(rows):
    curves = {}
    max_turn = max(int(t) for row in rows for arm in ARMS for t in row[arm]["turns"])
    for turn in range(1, max_turn + 1):
        sides = {arm: [eot(row[arm], turn) for row in rows if eot(row[arm], turn)] for arm in ARMS}
        common = [row for row in rows if all(eot(row[arm], turn) for arm in ARMS)]
        result = {}
        for arm, stages in sides.items():
            if not stages:
                result[arm] = dict(n=0)
                continue
            actions = set(k for row in stages for k in row["counts"])
            result[arm] = dict(n=len(stages), m1=median(row["strength"] for row in stages),
                m2=mean(row["tier"] for row in stages), health=mean(row["health"] for row in stages),
                remaining_gold=mean(row["gold"] for row in stages), decisions=mean(row["actions"] for row in stages),
                action_mean={k:mean(row["counts"].get(k,0) for row in stages) for k in sorted(actions)})
        result["common_m1_log_delta"] = matched(common, lambda a: math.log1p(eot(a,turn)["strength"]))
        result["common_m2_delta"] = matched(common, lambda a: eot(a,turn)["tier"])
        curves[turn] = result
    return curves


def normalizer():
    card_db = json.loads((ROOT/"engine/crates/hsbg-engine/data/cards.json").read_bytes())["cards"]
    cards = {card["id"]: card for card in card_db}
    def normalize(card):
        return cards[card].get("normal") or card
    return cards, normalize


def consecutive(times):
    return any(turn+1 in times for turn in times)


def build_sets(arm, normalize):
    return {int(t): {normalize(u[0]) for u in value["board"]} for t, value in arm["turns"].items()
            if int(t) >= 8}


def for_tribe(rows, tribe):
    if tribe is None:
        return rows
    mask = 1 << TYPES.index(tribe)
    return [row for row in rows if row["r1"]["lobby"] & mask]


def lineups(rows, normalize):
    def timestamps(arm, ids):
        return {turn for turn, cards in build_sets(arm, normalize).items() if set(ids) <= cards}
    checks = {}
    for name, (tribe, ids) in MOTIFS.items():
        cohort = for_tribe(rows, tribe)
        checks[name] = dict(single_board=rates(cohort, lambda a: bool(timestamps(a, ids))),
                            sustained=rates(cohort, lambda a: consecutive(timestamps(a, ids))))
    signals = {
        "positive_arcane_absorption_two_turns": ("ELEMENTAL",
             lambda a: len({turn for turn, gain, shop in a["absorptions"] if gain > 0 and shop > 0}) >= 2),
        "hogrider_gem_returns_two_turns": ("QUILBOAR",
             lambda a: len({g["turn"] for g in a["gems"] if g["immediate_stats"] > 0}) >= 2),
        "robot_drone_with_magnetic": ("MECHANICAL",
             lambda a: consecutive(timestamps(a, MOTIFS["robot_drone"][1])) and len(set(a["magnet"])) >= 2),
        "hope_poet_with_growth": ("DRAGON",
             lambda a: consecutive(timestamps(a, MOTIFS["hope_poet"][1])) and bool(a["hope_growth"])),
    }
    mechanisms = {name: rates(for_tribe(rows, tribe), predicate) for name, (tribe,predicate) in signals.items()}
    for key, tribe in (("element", "ELEMENTAL"), ("pig","QUILBOAR")):
        cohort = for_tribe(rows,tribe)
        jointly_offered = [r for r in cohort if all(key in r[a]["offers"] for a in ARMS)]
        mechanisms[key+"_offered"] = rates(cohort,lambda a: key in a["offers"])
        mechanisms[key+"_buy_given_both_offered"] = rates(jointly_offered,lambda a: key in a["buys"])
    return dict(motifs=checks, mechanisms=mechanisms)


def economy(rows):
    fields = ("buy", "sell", "refresh", "upgrade", "play", "magnetize", "move", "freeze")
    cohorts = {}
    for begin,end in ((1,7),(8,10),(11,13)):
        available=[r for r in rows if all(eot(r[a],t) for a in ARMS for t in range(begin,end+1))]
        def values(arm, kind):
            stages=[eot(arm,t) for t in range(begin,end+1)]
            if kind=="actions":
                return mean(t["actions"] for t in stages)
            if kind in ("remaining_gold","hand_n","health"):
                key="gold" if kind=="remaining_gold" else kind
                return mean(t[key] for t in stages)
            return mean(t["counts"].get(kind,0) for t in stages)
        measures={}
        if available:
            for key in ("actions","remaining_gold","hand_n","health",*fields):
                measures[key]={a:mean(values(r[a],key) for r in available) for a in ARMS}
                measures[key]["delta"]=matched(available,lambda arm: values(arm,key))
        cohorts[f"{begin}-{end}"]={"n":len(available),"measures":measures}
    return cohorts


def prevalent(arm, catalog, normalize):
    given = max(arm["turns"], key=int)
    if int(given) < 9:
        return ()
    counts = Counter(race for unit in arm["turns"][given]["board"]
                     for race in catalog[normalize(unit[0])].get("races", ()) if race in TYPES)
    return tuple(sorted(race for race, count in counts.items() if count >= 4))


def paired_exploration(panels, normalize):
    occurrence = {}
    for panel, rows in panels.items():
        fields = {a: {} for a in ARMS}
        for row in rows:
            for arm in ARMS:
                turns = build_sets(row[arm], normalize)
                retained = set()
                for turn in turns.keys() & {t-1 for t in turns}:
                    retained.update(combinations(sorted(turns[turn] & turns[turn+1]), 2))
                fields[arm][row["seed"]] = retained
        occurrence[panel] = fields
    results = []
    for ids, total in Counter(ids for values in occurrence["hold"]["final"].values() for ids in values).items():
        if total < 5:
            continue
        counts = {panel:{arm:sum(ids in v for v in fields[arm].values()) for arm in ARMS}
                  for panel, fields in occurrence.items()}
        delta = {p:v["final"]-v["r1"] for p,v in counts.items()}
        if delta["hold"] >= 3 and delta["human"] >= 2:
            results.append(dict(cards=ids,counts=counts,
                delta={p:estimate([int(ids in fields["final"][seed])-int(ids in fields["r1"][seed])
                                  for seed in fields["r1"]]) for p,fields in occurrence.items()}))
    return sorted(results, key=lambda x: -sum(x["delta"][p]["mean"] for p in panels))


def opening_stats(path):
    results = {}
    rows = json.loads(path.read_bytes())["rows"]
    for case in sorted({r["case"] for r in rows}):
        cohort = [r for r in rows if r["case"] == case]
        def times(arm):
            if case.startswith("elem"):
                return [t["turn"] for t in arm if any(e["shop"] > 0 and e["gain"] > 0
                                                     for e in t["absorptions"])]
            return [t["turn"] for t in arm if any(e["immediate_stats"] > 0 for e in t["choices"])]
        strength = {a: median(r[a][-1]["strength"] for r in cohort) for a in ARMS}
        results[case] = dict(p50_t13=strength, at_least_two_payout_turns=rates(cohort, lambda a: len(times(a)) >= 2),
                             log_t13_delta=matched(cohort, lambda a: math.log1p(a[-1]["strength"])))
    return results


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("out", type=Path)
    cli.add_argument("--openings", type=Path)
    args = cli.parse_args()
    by_panel = {"hold": [], "human": []}
    with gzip.open(args.source, "rt", encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            by_panel[row["panel"]].append(row)
    catalog, normalize = normalizer()
    report = {}
    for panel, rows in by_panel.items():
        assert rows and len({r["seed"] for r in rows}) == len(rows)
        assert all(r["r1"]["lobby"] == r["final"]["lobby"] for r in rows)
        same_hero = [r for r in rows if all(eot(r[a],1) for a in ARMS) and
                     eot(r["r1"],1)["hero"] is not None and eot(r["r1"],1)["hero"]==eot(r["final"],1)["hero"]]
        report[panel] = dict(scores=pair_stats(rows), turns=turn_stats(rows), economy=economy(rows),
                             identical_hero=dict(n=len(same_hero),rank_delta=matched(same_hero,lambda a:a["rank"])),
                             lineups=lineups(rows, normalize),
                             dominant_tribes={race:rates(for_tribe(rows, race),
                                 lambda a: race in prevalent(a,catalog,normalize)) for race in TYPES})
    explored = paired_exploration(by_panel, normalize)
    for entry in explored:
        entry["names"] = [catalog[card]["name"] for card in entry["cards"]]
    report["exploratory_pairs"] = explored
    if args.openings:
        report["controlled_openings"] = opening_stats(args.openings)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
