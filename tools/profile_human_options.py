"""Check availability of verified human choices and two-turn examples before planning a teacher."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.profile_lineup_use import read_cards


def profile():
    root=Path('data/real_library/firestone-20261112-1000')
    db=read_cards()
    meta={}
    for line in (root/'library.jsonl').open(encoding='utf8'):
        r=json.loads(line)
        if r['perspective']=='self':
            meta[r['game_id']]=(r['split'],str(r.get('archetype')),r['build_number'])
    splits=defaultdict(Counter)
    games=defaultdict(set)
    turns=defaultdict(lambda:defaultdict(set))
    tribes=defaultdict(set)
    def norm(cid):
        if cid not in db:
            return None
        return db[cid].get('normal') or cid
    with gzip.open(root/'actions.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            a=json.loads(line)
            if a['game_id'] not in meta or not 7<=a['turn']<=12 or a['action_type'] not in ('sell','buy_minion'):
                continue
            game=a['game_id']
            split,archetype,build=meta[game]
            if build!=a['build_number']:
                raise ValueError('observation versus action: version mismatch')
            kind=a['action_type']
            seq=a['shop_cards'] if kind=='buy_minion' else a['board_cards']
            slot=a['shop_slot'] if kind=='buy_minion' else a['board_slot']
            counts=splits[split]
            counts[kind+'_all']+=1
            cid=norm(a['card_id'])
            if (cid is None or type(slot)!=int or not 0<=slot<len(seq) or
                    seq[slot]!=a['card_id'] or any(norm(x) is None for x in seq) or
                    len({norm(x) for x in seq if db[x]['kind']=='minion'})<2):
                continue
            if kind=='sell' and len(seq)!=7:
                continue
            counts[kind+'_identified']+=1
            games[split].add(game)
            turns[split][game].add((a['turn'],kind))
            tribes[split].add(archetype)
    for split in splits:
        splits[split]['games']=len(games[split])
        splits[split]['archetype_tags']=len(tribes[split])
        splits[split]['games_with_buy_and_full_board_sell_in_two_adjacent_turns']=sum(
            any(all((t,k) in events for k in ('sell','buy_minion') for t in (turn,turn+1))
                for turn in range(7,12)) for events in turns[split].values())
    return dict(source_actions=file_identity(root/'actions.jsonl.gz'),source_boards=file_identity(root/'library.jsonl'),
                splits=dict(splits),source_builds=dict(Counter(m[2] for m in meta.values())),
                status='Proposal data only. Exact chosen card/slot and alternative card identities; no native mask, '
                       'no assumed counterfactual outcome, no raw identifiers exported.')


if __name__=='__main__':
    atomic_json(Path(sys.argv[1]),profile())
