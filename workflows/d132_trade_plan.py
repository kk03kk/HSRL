"""Fit a two-turn trade proposal with a next-board auxiliary target and a history ablation."""
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.common import engine_info
from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from workflows.human_trade_options import Cards, MODEL, FIELDS
from workflows.d132_ordered_replays import LIB

ORDERED = Path('artifacts/runs/nus-d132-ordered-20261118-1')
FIT = Path('artifacts/runs/nus-d132-fit-20261118-1')
HISTORY_WIDTH = 12
KINDS = ('buy_minion', 'sell', 'play_minion')


def provenance():
    plan = json.loads(Path('config/research/natural12m-20261118.json').read_text())
    src = source_identity(Path.cwd())
    if (src['tracked_dirty'] or engine_info()['source_hash'] != plan['engine_fingerprint'] or
            any(file_identity(k)['sha256'] != v for k,v in plan['asset_sha256'].items())):
        raise ValueError('source, engine, or asset drift')
    return dict(revision=src, inputs=plan['asset_sha256'], engine=engine_info(), live_pin=plan['live_pin_sha256'])


class Representation(Cards):
    def __init__(self, policy):
        import hsbg_engine as engine
        super().__init__()
        net = policy.evaluator.model
        with torch.inference_mode():
            vectors = net._card_repr(torch.arange(net.n_cards)[None])[0].numpy()
        vectors /= np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True), 1e-8)
        self.sim = vectors @ vectors.T
        self.native = {engine.card_info(i)[0]:i for i in range(1,net.n_cards)}
        self.reverse = {i:k for k,i in self.native.items()}

    def features(self, k, board, hand, turn, gold, selling, history):
        # Centre history on the current and preceding recruit turn. Never read later actions.
        row = self.describe(k, board, hand, set(), selling, min(turn,12))
        row.pop(8)  # Describe's just-played flag is replaced by the recoverable ordered history below.
        i = self.native.get(k, 0)
        for group in (board, hand):
            sims = [float(self.sim[i,self.native[q]]) for q in group if q in self.native]
            row.extend((max(sims,default=0), sum(sims)/max(1,len(sims))))
        row.extend((min(gold,20)/10, float(selling), len(board)/7, len(hand)/10))
        for t in (turn-1,turn):
            for kind in KINDS:
                choices = history.get((t,kind), ())
                row.extend((min(3,choices.count(self.base[k]))/3,
                            min(5,sum(bool(self.tribes[k] & self.tribes.get(q,set())) for q in choices))/5))
        return row


class TradeHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(width,32), nn.Tanh(),nn.Linear(32,16),nn.Tanh())
        self.choice = nn.Linear(16,1)
        self.next_board = nn.Linear(16,1)

    def forward(self,x,sell):
        h = self.shared(x)
        future = self.next_board(h).squeeze(-1)
        return self.choice(h).squeeze(-1) + (1-2*sell)*future, future


def examples(cards):
    boards = {}
    for line in (LIB/'library.jsonl').open():
        r = json.loads(line)
        if r['perspective']=='self':
            boards[r['game_id'],r['turn']] = {cards.base.get(m['card_id']) for m in r['board_detail']}
    rows = []
    for path in sorted(ORDERED.glob('*.json.gz')):
        with gzip.open(path,'rt') as f:
            record=json.load(f)
        split,tag = record['meta']
        game=path.name.split('.')[0]
        history=defaultdict(list)
        for event in record['events']:
            kind,turn,k=event['kind'],event['turn'],event['card']
            if k not in cards.base:
                continue
            if turn>=7 and kind in KINDS[:2]:
                slots=list(dict.fromkeys(q for q in event['slots'] if q in cards.db and cards.db[q]['kind']=='minion'))
                if k in slots and len({cards.base[q] for q in slots})>=2 and all(q in cards.db for q in event['slots']):
                    selling=kind=='sell'
                    x=[cards.features(q,event['board'],event['hand'],turn,event['gold'],selling,history) for q in slots]
                    next_board=boards.get((game,turn+1))
                    y=[-1 if next_board is None else int(cards.base[q] in next_board) for q in slots]
                    cohort='unseen_demon' if 'demon' in tag.lower() else split
                    rows.append((game,cohort,kind,x,slots.index(k),y))
            history[turn,kind].append(cards.base[k])
    return rows


def train(rows,out):
    n=len(rows)
    width=len(rows[0][3][0])
    slots=max(len(r[3]) for r in rows)
    x=torch.zeros(n,slots,width)
    valid=torch.zeros(n,slots,dtype=torch.bool)
    targets=torch.full((n,slots),-1.)
    for i,r in enumerate(rows):
        m=len(r[3])
        x[i,:m]=torch.tensor(r[3])
        valid[i,:m]=True
        targets[i,:m]=torch.tensor(r[5])
    label=torch.tensor([r[4] for r in rows])
    sell=torch.tensor([float(r[2]=='sell') for r in rows])[:,None]
    fit=np.array([i for i,r in enumerate(rows) if r[1]=='train'])
    counts=Counter(rows[i][0] for i in fit)
    weights=torch.tensor([1/max(1,counts[r[0]]) for r in rows])
    weights/=weights[fit].mean()
    reports={}
    for history in (False,True):
        torch.manual_seed(13207)
        head=TradeHead(width)
        features=x.clone()
        if not history:
            features[:,:,-HISTORY_WIDTH:]=0
        opt=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.01)
        rng=np.random.default_rng(13207)
        for epoch in range(12):
            for batch in np.array_split(rng.permutation(fit),int(np.ceil(len(fit)/256))):
                logits,future=head(features[batch],sell[batch])
                ce=nn.functional.cross_entropy(logits.masked_fill(~valid[batch],-1e9),label[batch],reduction='none')
                available=targets[batch]>=0
                aux=nn.functional.binary_cross_entropy_with_logits(future,targets[batch].clamp_min(0),reduction='none')
                loss=((ce+.25*(aux*available).sum(1)/available.sum(1).clamp_min(1))*weights[batch]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.inference_mode():
            logits,_=head(features,sell)
            logits=logits.masked_fill(~valid,-1e9)
            hit=(logits.argmax(1)==label).numpy()
            ce=nn.functional.cross_entropy(logits,label,reduction='none').numpy()
        tag='history' if history else 'masked'
        report={}
        for cohort in ('train','evaluation','unseen_demon'):
            for kind in KINDS[:2]:
                per_game=defaultdict(list)
                for i,r in enumerate(rows):
                    if r[1:3]==(cohort,kind):
                        per_game[r[0]].append(i)
                report[cohort+'/'+kind]=dict(games=len(per_game),rows=sum(map(len,per_game.values())),
                    hit=float(np.mean([hit[g].mean() for g in per_game.values()])),
                    nll=float(np.mean([ce[g].mean() for g in per_game.values()])))
        reports[tag]=report
        torch.save(dict(width=width,state=head.state_dict()),out/(tag+'.pt'))
    return reports


if __name__=='__main__':
    torch.set_num_threads(1)
    FIT.mkdir(parents=True,exist_ok=False)
    verified=provenance()
    raw=json.loads((ORDERED/'complete.json').read_text())
    if raw['cohort']['train']!=742:
        raise ValueError('incomplete demonstrations')
    policy=make_eval_policy(MODEL,protocol=EvalProtocol(mirror_snaps=FIELDS['development']),device='cpu')['policy']
    reports=train(examples(Representation(policy)),FIT)
    atomic_json(FIT/'complete.json',dict(**verified,source_report=file_identity(ORDERED/'complete.json'),
        scores=reports,history=file_identity(FIT/'history.pt'),masked=file_identity(FIT/'masked.pt')))
