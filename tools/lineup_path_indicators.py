"""Count observed switching, reinvestment and contextual purchases on a sealed panel."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.profile_lineup_use import read_cards
from tools.compare_research_panels import load_sealed_panel


def switch(tribes, turns, db):
    for old_end in sorted(tribes):
        if old_end < 7 or old_end-1 not in tribes:
            continue
        old = tribes[old_end-1] & tribes[old_end]
        for end in range(old_end+2,max(tribes)+1):
            new = tribes[end-1] & tribes[end]
            for x,n in old.items():
                for y,k in new.items():
                    if n<3 or k<3 or x==y:
                        continue
                    if max(tribes[t][x] for t in (end-1,end))>1 or max(tribes[t][y] for t in (old_end-1,old_end))>1:
                        continue
                    actions = [a for t in range(old_end+1,end+1) for a in turns[t]["actions"]]
                    sold = any(16<=a[0]<23 and x in db[a[2]]["races"] for a in actions)
                    played = {db[a[2]].get("normal") or a[2] for a in actions if
                              46<=a[0]<126 and a[1]=="Minion" and y in db[a[2]]["races"]}
                    if sold and len(played)>=2:
                        return True
    return False


def reinvest(actions):
    count = 0
    for i,(buy,play) in enumerate(zip(actions,actions[1:])):
        if not (8<=buy[0]<16 and 46<=play[0]<126 and buy[2]==play[2]=="BG33_815"):
            continue
        cost = buy[4]-buy[5]
        if 0<cost<play[5]-play[4] and any(a[0]!=0 and a[5]<a[4] for a in actions[i+2:]):
            count += 1
    return count


def indicators(panel, ledger):
    db = read_cards()
    normal = {k:c.get("normal") or k for k,c in db.items()}
    rosters = defaultdict(dict)
    for m in panel["models"]:
        for row in m["trace"]:
            rosters[m["name"],row["seed"]][row["turn"]] = row["roster"]
    result = defaultdict(Counter)
    seen = set()
    with gzip.open(ledger,"rt") as stream:
        for line in stream:
            game = json.loads(line)
            key = game["model"],game["seed"]
            if key in seen or key not in rosters:
                raise ValueError("invalid seed join")
            seen.add(key)
            rs = rosters[key]
            turns = {int(t):v for t,v in game["turns"].items()}
            tribes = {t:Counter(r for c in board for r in db[c["card_id"]]["races"] if r!="ALL")
                      for t,board in rs.items()}
            counts = result[game["model"]]
            if switch(tribes,turns,db):
                counts["switch"] += 1
                counts["switch_rank_"+str(game["rank"])] += 1
            trades = sum(reinvest(v["actions"]) for v in turns.values())
            counts["profitable_bounty_reinvest"] += int(trades>0)
            counts["repeated_profitable_bounty_reinvest"] += int(trades>=2)
            for t in range(8,13):
                if t not in turns:
                    continue
                previous = rs.get(t-1,[])
                shields = sum(bool((c["keywords"] or 0)&2) for c in previous)
                rogers = any(normal[c["card_id"]]=="BG33_823" for c in previous)
                offers = {normal[c] for c in turns[t]["offers"]}
                bought = {normal[a[2]] for a in turns[t]["actions"] if 8<=a[0]<16}
                for card,condition in (("BG36_364",shields>=2),("BG33_825",rogers)):
                    if card in offers:
                        prefix = card+("_with_" if condition else "_without_")
                        counts[prefix+"offered_turns"] += 1
                        counts[prefix+"bought_turns"] += int(card in bought)
    if seen!=set(rosters):
        raise ValueError("missing games")
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel",type=Path)
    parser.add_argument("ledger",type=Path)
    args = parser.parse_args()
    print(json.dumps(indicators(load_sealed_panel(args.panel),args.ledger),indent=2))
