"""汇总冻结评测与里程碑，不加载模型。"""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median

def strength_quantiles(values):
    """对每张战团总攻血取分位；不能相加边际中位数。"""
    values = sorted(values)
    if not values:
        return {}
    def q(p):
        position=(len(values)-1)*p; lo=int(position); hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(position-lo)
    return {"p25_strength":q(.25),"p50_strength":median(values),"p75_strength":q(.75),"p90_strength":q(.9)}



def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def by_turn(report, turns=None):
    if not report["valid"]:
        raise ValueError("评测未通过工程门禁")
    seats = report["per_seat"]
    games = sum(len(seat["rank_sequence"]) for seat in seats)
    if not games or any("end_turn_economy" not in seat or "combat_exposure" not in seat for seat in seats):
        raise ValueError("缺少经营或交战数据")
    if turns is None:
        last = max((r["turn"] for seat in seats for key in ("end_turn_economy", "combat_exposure")
                    for r in seat[key]), default=0)
        turns = range(1, max(19, last) + 1)
    result = {}
    for turn in turns:
        economy = [r for seat in seats for r in seat["end_turn_economy"] if r["turn"] == turn]
        battles = [r for seat in seats for r in seat["combat_exposure"] if r["turn"] == turn]
        if len(economy) > games or len(battles) > games:
            raise ValueError(f"第 {turn} 回合记录超过评测局数")
        live = [r for r in battles if not r["ghost"]]
        row = dict(games=games, reached_end_turn=len(economy), combats=len(battles),
                   reached_end_turn_rate=len(economy)/games,
                   live_combats=len(live), live_wins=sum(r["win"] for r in live),
                   live_ties=sum(r["tie"] for r in live), ghost_combats=len(battles)-len(live),
                   live_win_rate=mean(r["win"] for r in live) if live else None,
                   live_loss_rate=mean(not r['win'] and not r['tie'] for r in live) if live else None,
                   mean_damage_dealt=mean(r['damage'] if r['win'] else 0 for r in battles) if battles else None,
                   mean_damage_taken=mean(r["damage_taken"] for r in battles) if battles else None)
        if economy:
            row.update(strength_quantiles([r["attack"]+r["board_hp"] for r in economy]))
            eot_rows = [r for r in economy if r.get('eot_available')]
            if any('eot_available' in r for r in economy):
                row['eot_observations'] = len(eot_rows)
                row['eot_missing'] = len(economy)-len(eot_rows)
                row.update({'eot_'+k:v for k,v in strength_quantiles(
                    [r['eot_attack']+r['eot_board_hp'] for r in eot_rows]).items()})
            row["p50_tier"] = median(r["tier"] for r in economy)
            row["tier4_plus_rate"] = mean(r["tier"] >= 4 for r in economy)
            row["tier5_plus_rate"] = mean(r["tier"] >= 5 for r in economy)
            row["tier6_rate"] = mean(r["tier"] >= 6 for r in economy)
            row.update(mean_tier=mean(r["tier"] for r in economy),
                       tier_histogram=dict(sorted(Counter(r["tier"] for r in economy).items())),
                       mean_actions=mean(r["actions"] for r in economy),
                       mean_gold_left=mean(r["gold"] for r in economy),
                       mean_hero_health=mean(r["health"] for r in economy),
                       forced_end_turns=sum(r["forced"] for r in economy),
                       mean_attack=mean(r["attack"] for r in economy),
                       mean_board_hp=mean(r["board_hp"] for r in economy),
                       mean_strength=mean(r["attack"] + r["board_hp"] for r in economy),
                       median_attack=median(r["attack"] for r in economy),
                       median_board_hp=median(r["board_hp"] for r in economy))
            if all("action_counts" in r for r in economy):
                counts = Counter()
                for event in economy:
                    counts.update(event["action_counts"])
                available = sum(r.get("upgrade_legal_steps", 0) > 0 for r in economy)
                taken = sum(r["action_counts"].get("upgrade", 0) > 0 for r in economy)
                row.update(action_counts=dict(counts),
                           mean_actions_by_type={k: v/len(economy) for k, v in sorted(counts.items())},
                           consecutive_repeats=sum(r["consecutive_repeats"] for r in economy),
                           upgrade_available_turns=available, upgrade_taken_turns=taken,
                           upgrade_taken_per_available_turn=taken/available if available else None)
            if all('board' in r for r in economy):
                row['mean_board_count']=mean(len(r['board']) for r in economy)
                row['gold_net_spent_during_actions']=mean(sum(max(0,-e['net_change'])
                    for e in r.get('gold_events',[])) for r in economy)
                row['gold_net_gained_during_actions']=mean(sum(max(0,e['net_change'])
                    for e in r.get('gold_events',[])) for r in economy)
        result[turn] = row
    return result


def summarize(run, turns):
    report = read(run / "report.json")
    stages = []
    for stage in report["stages"]:
        pair = stage.get("paired", {}).get("hard")
        stages.append(dict(target=stage["target"], version=stage["checkpoint"]["version"],
                           paired=({k: pair[k] for k in ("n", "protocol_sha256", "delta", "ci_low", "ci_high")}
                                   if pair else None)))
    final = report["final_evaluations"]["latest"]["hard"]
    models = (("reference", final["before_sha256"]), ("latest", final["after_sha256"]))
    matches = {}
    for path in sorted(run.glob("baselines/hard-*.json")) + sorted(run.glob("final/*/hard.json")):
        evaluation = read(path)
        if evaluation["protocol_sha256"] != final["protocol_sha256"]:
            continue
        sha = evaluation["checkpoint"]["sha256"]
        for label, expected in models:
            if sha == expected:
                matches[label] = by_turn(evaluation, turns)
    if set(matches) != {"reference", "latest"}:
        raise ValueError("缺少同协议终评原始文件")
    return dict(stages=stages, final_protocol_sha256=final["protocol_sha256"],
                economy_stage="before_end_turn", final_by_turn=matches)


def panel_diagnostics(report, path):
    """保留总榜、席位结果和每回合经营统计，原始记录单独引用。"""
    return dict(
        checkpoint_sha256=report["checkpoint"]["sha256"],
        protocol_sha256=report["protocol_sha256"],
        mean_rank=report["mean_rank"], mean_rank_ci95=report.get("mean_rank_ci95"),
        top4_rate=report["top4_rate"], win_rate=report["win_rate"],
        ranks_by_seat=[seat["rank_sequence"] for seat in report["per_seat"]],
        strength_primary_statistic="P50(sum_attack+sum_health)",
        raw_evaluation=str(path), economy_stage="before_end_turn", turns=by_turn(report),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--turns", nargs="+", type=int, default=[8, 10, 12, 15])
    args = parser.parse_args()
    print(json.dumps(summarize(args.run, args.turns), ensure_ascii=False, indent=2))
