"""Recover chronological, entity- and payment-checked trade opportunities from raw Firestone XML."""
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json, file_identity
from workflows.parse_firestone_replay_xml import (
    BLOCK_PLAY, CARD_KINDS, TAG_CONTROLLER, ZONE_PLAY, ZONE_SETASIDE, CardDb, XmlReplay,
)

ROOT = Path('data/firestone_replays/refresh-20261112T031450Z')
LIB = Path('data/real_library/firestone-20261112-1000')
CARDS = 'engine/crates/hsbg-engine/data/cards.json'


class OrderedReplay(XmlReplay):
    def __init__(self, db):
        super().__init__(db)
        self.payments = Counter()

    def balance(self):
        tags = self.ent.get(str(self.main), {}).get('tags', {})
        if '26' not in tags:
            return None
        return int(tags['26']) + int(tags.get('295', 0)) - int(tags.get('25', 0))

    def _on_block_start(self, el):
        # Capture before the first nested tag update. Only recruit decisions of the focal player.
        if el.get('type') != BLOCK_PLAY or not 6 <= self.round <= 12:
            self.block_stack.append(dict(eid=el.get('entity'), bt=el.get('type'), action=None))
            return
        actor = self.ent.get(el.get('entity'), {})
        if actor.get('tags', {}).get(TAG_CONTROLLER) != self.main_ctrl or self.main_ctrl is None:
            self.block_stack.append(dict(eid=el.get('entity'), bt=el.get('type'), action=None))
            return
        super()._on_block_start(el)
        a = self.block_stack[-1]['action']
        if not a or a['action_type'] not in ('play_minion', 'sell', 'buy_minion'):
            return
        a['seq'] = self.seq
        a['gold_before'] = self.balance()
        if a['action_type'] == 'play_minion':
            return
        sell = a['action_type'] == 'sell'
        board = self._board(self.main_ctrl)
        if sell:
            candidates = board
        else:
            candidates = []
            seen = set()
            for zone in (ZONE_PLAY, ZONE_SETASIDE):
                for item in self._cards(zone, self.bob_ctrl, CARD_KINDS) if self.bob_ctrl else ():
                    key = (item['position'], item['card_id'])
                    if key not in seen:
                        candidates.append(item)
                        seen.add(key)
            candidates.sort(key=lambda x: (x['position'], x['eid']))
            candidates = candidates[:8]
        slot = a['board_slot'] if sell else a['shop_slot']
        if not isinstance(slot, int) or slot >= len(candidates):
            return
        target = candidates[slot]
        if target['eid'] != a['target_id'] or target['card_id'] != a['card_id']:
            return
        a['slots'] = [item['card_id'] for item in candidates]
        a['slot'] = slot
        a['entity'] = target['eid']

    def _on_block_end(self, el):
        old = len(self.actions)
        super()._on_block_end(el)
        if len(self.actions) > old:
            self.actions[-1]['gold_after'] = self.balance()

    def result(self):
        events = []
        for a in sorted((x for x in self.actions if 'seq' in x), key=lambda x: x['seq']):
            k = a['action_type']
            if k == 'play_minion':
                events.append(dict(turn=a['turn'], seq=a['seq'], kind=k, card=a['card_id']))
                continue
            pre, post = a['gold_before'], a['gold_after']
            delta = post-pre if pre is not None and post is not None else None
            self.payments[f'{k}/{delta}'] += 1
            # Only unambiguous standard-price trades, with the selected entity in its exact slot.
            if ('slots' not in a or delta != (1 if k == 'sell' else -3) or pre < 0 or
                    (k == 'sell' and len(a['board_cards']) != 7)):
                continue
            events.append(dict(turn=a['turn'], seq=a['seq'], kind=k, card=a['card_id'],
                               gold=pre, entity=a['entity'], slot=a['slot'], slots=a['slots'],
                               board=a['board_cards'], hand=a['hand_cards']))
        return events


def extract(args):
    game, meta, expected, dest = args
    path = ROOT/'xmlz'/(game + '.xml.zip')
    if file_identity(path)['sha256'] != expected:
        raise ValueError('raw replay checksum mismatch')
    parser = OrderedReplay(CardDb(CARDS))
    events = parser.parse(path)
    with gzip.open(dest/(game+'.json.gz'), 'xt', encoding='utf8') as f:
        json.dump(dict(meta=meta, events=events), f, separators=(',', ':'))
    return dict(events=len(events), payments=dict(parser.payments))


def main(out):
    from workflows.human_trade_options import ASSETS, Cards
    out.mkdir(parents=True, exist_ok=False)
    if any(file_identity(LIB/k)['sha256'] != v for k,v in ASSETS.items()):
        raise ValueError('human source changed')
    meta = {}
    for line in (LIB/'library.jsonl').open():
        row = json.loads(line)
        if row['perspective'] == 'self':
            if row['build_number'] != 255790:
                raise ValueError('mixed builds')
            meta[row['game_id']] = [row['split'], str(row.get('archetype'))]
    # Recover the original consecutive-turn cohort, before applying stricter XML checks.
    cards = Cards()
    turns = defaultdict(lambda: defaultdict(set))
    with gzip.open(LIB/'actions.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            k, t, card = r['action_type'], r['turn'], r['card_id']
            if k not in ('buy_minion','sell') or not 7 <= t <= 12 or card not in cards.db:
                continue
            seq, slot = (r['board_cards'],r['board_slot']) if k=='sell' else (r['shop_cards'],r['shop_slot'])
            if (not isinstance(slot,int) or not 0<=slot<len(seq) or seq[slot]!=card or
                    (k=='sell' and len(seq)!=7) or any(x not in cards.db for x in seq)):
                continue
            if len({cards.base[x] for x in seq if cards.db[x]['kind']=='minion'}) >= 2:
                turns[r['game_id']][t].add(k)
    selected = [g for g,t in turns.items() if g in meta and any(len(v)==2 and len(t.get(i+1,()))==2 for i,v in t.items())]
    hashes = {Path(p).name:h for h,p in (line.split(None,1) for line in (ROOT/'SHA256SUMS').read_text().splitlines())}
    counts = Counter(meta[g][0] for g in selected)
    if counts['train'] != 742:
        raise ValueError(f'cohort drift: {counts}')
    args = [(g,meta[g],hashes[g+'.xml.zip'],out) for g in sorted(selected)]
    with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK', 4))) as pool:
        rows = list(pool.map(extract,args))
    payments = Counter()
    for r in rows:
        payments.update(r['payments'])
    atomic_json(out/'complete.json',dict(cohort=counts, games=len(args), events=sum(r['events'] for r in rows),
                                        payments=dict(payments), raw_manifest=file_identity(ROOT/'SHA256SUMS')))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
