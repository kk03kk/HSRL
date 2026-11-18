"""Learn contextual preferences between real available minions; test only within native trade actions."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.economic_tasks import item_card
from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.profile_lineup_use import read_cards
from tools.assess_investment_timing import compare
from workflows.probe_investment_timing import TimingPolicy, MODEL, FIELDS


DATA = Path('data/real_library/firestone-20261112-1000')
ASSETS = {'actions.jsonl.gz':'2c4f5b6aa2b849d74020d9a24b2f4ad72e2b6f7caacf5dda087e254921cc6ded',
          'library.jsonl':'8e3446b8c4b16af7cd881201f29c8144e60357322cc9e0081f174e714cdcabc1'}
CACHE = {}


class Cards:
    def __init__(self):
        self.db = read_cards()
        self.base = {k:c.get('normal') or k for k,c in self.db.items()}
        self.tribes = {k:set(c['races'])-{'ALL'} for k,c in self.db.items()}
        self.ids = {k:i+1 for i,k in enumerate(sorted(self.db))}

    def describe(self, k, board, hand, played, sell, turn):
        info = self.db[k]
        owned = list(board)
        if sell:
            owned.remove(k)
        tribe = self.tribes[k]
        board_match = sum(bool(tribe & self.tribes.get(q,set())) for q in owned)
        hand_match = sum(bool(tribe & self.tribes.get(q,set())) for q in hand)
        copies = sum(self.base.get(q)==self.base[k] for q in owned+list(hand))
        kw = info['keywords']
        tier = info['tier']/6
        return [tier,np.log1p(max(0,info['atk']+info['hp']))/4,int(bool(info.get('normal'))),
                board_match/6,hand_match/5,min(copies,3)/3,int(copies==1),int(copies>=2),
                int(self.base[k] in played),int('BATTLECRY' in kw),int('DEATHRATTLE' in kw),
                int('DIVINE_SHIELD' in kw),int('MAGNETIC' in kw),int('AVENGE' in kw),
                tier*(turn-7)/5,board_match*(turn-7)/30,
                int('BATTLECRY' in kw)*int(self.base[k] in played)]


def examples(cards):
    meta = {}
    for line in (DATA/'library.jsonl').open(encoding='utf8'):
        r = json.loads(line)
        if r['perspective']=='self':
            meta[r['game_id']] = (r['split'],str(r.get('archetype')),r['build_number'])
    out = {'buy_minion':[],'sell':[]}
    played = {}
    with gzip.open(DATA/'actions.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            a = json.loads(line)
            game,turn,k,kind = a['game_id'],a['turn'],a['card_id'],a['action_type']
            if game not in meta or not 7<=turn<=12 or not k in cards.db:
                continue
            history = played.setdefault((game,turn),set())
            if kind=='play_minion':
                history.add(cards.base[k])
            if kind not in out:
                continue
            sell = kind=='sell'
            board,hand = a['board_cards'],a['hand_cards']
            seq,slot = (board,a['board_slot']) if sell else (a['shop_cards'],a['shop_slot'])
            if (type(slot)!=int or not 0<=slot<len(seq) or seq[slot]!=k or
                    any(q not in cards.db for q in seq) or (sell and len(board)!=7)):
                continue
            options = list(dict.fromkeys(q for q in seq if cards.db[q]['kind']=='minion'))
            if len({cards.base[q] for q in options})<2 or k not in options:
                continue
            split,archetype,build = meta[game]
            if build!=255790 or a['build_number']!=build:
                raise ValueError('mixed replay builds')
            demon = 'demon' in archetype.lower()
            x = [cards.describe(q,board,hand,history,sell,turn) for q in options]
            out[kind].append((game,split,demon,[cards.ids[q] for q in options],x,options.index(k)))
    return out


def fit(rows, vocab):
    import torch
    n = len(rows)
    width = max(len(r[3]) for r in rows)
    dim = len(rows[0][4][0])
    ids = torch.zeros((n,width),dtype=torch.long)
    x = torch.zeros(n,width,dim)
    target = torch.tensor([r[5] for r in rows])
    for i,r in enumerate(rows):
        ids[i,:len(r[3])] = torch.tensor(r[3])
        x[i,:len(r[4])] = torch.tensor(r[4])
    train = torch.tensor([r[1]=='train' and not r[2] for r in rows])
    freq = Counter(r[0] for r in rows if r[1]=='train' and not r[2])
    iw = torch.tensor([1/max(1,freq[r[0]]) for r in rows])[train]
    iw /= iw.sum()
    train_x,train_ids,train_y = x[train],ids[train],target[train]
    out, scores = {},{}
    for context in (False,True):
        coef = torch.zeros(dim,requires_grad=context)
        bias = torch.zeros(vocab+1,requires_grad=True)
        opt = torch.optim.LBFGS([bias]+([coef] if context else []),max_iter=55,line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            logits = (bias[train_ids]+train_x@coef).masked_fill(train_ids==0,-1e8)
            loss = (torch.nn.functional.cross_entropy(logits,train_y,reduction='none')*iw).sum()
            loss = loss + .04*bias.square().mean()+.001*coef.square().sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            logits = (bias[ids]+x@coef).masked_fill(ids==0,-1e8)
            nll = torch.nn.functional.cross_entropy(logits,target,reduction='none').numpy()
            hit = (logits.argmax(1)==target).numpy()
        label = 'context' if context else 'card_only'
        scores[label] = {}
        for split,group in [('evaluation',False),('unseen_demon',True)]:
            grouped = {}
            for i,r in enumerate(rows):
                if (r[2] if group else r[1]=='evaluation' and not r[2]):
                    grouped.setdefault(r[0],[]).append(i)
            if not grouped:
                raise ValueError('empty human test cohort')
            scores[label][split] = dict(games=len(grouped),rows=sum(map(len,grouped.values())),
                nll=float(np.mean([nll[g].mean() for g in grouped.values()])),
                accuracy=float(np.mean([hit[g].mean() for g in grouped.values()])))
        if context:
            out = dict(coef=coef.tolist(),train_games=len(freq),train_rows=int(train.sum()))
    return dict(**out,scores=scores)


class Trade(TimingPolicy):
    def __init__(self, parent, cards, weights, enabled):
        super().__init__(parent,'parent',False)
        self.cards,self.weights,self.enabled = cards,weights,enabled
        self.played = set()
        self.adjustments = Counter()

    def cid(self, item):
        import hsbg_engine
        idx = item_card(item)
        if idx not in CACHE['names']:
            CACHE['names'][idx] = hsbg_engine.card_info(idx+1)[0]
        return CACHE['names'][idx]

    def decide(self, game, seat):
        if self.turn!=int(game.turn):
            self.played.clear()
        dec = super().decide(game,seat)
        if 7<=game.turn<=12 and 46<=dec.action<126:
            item=self.player['hand'][(dec.action-46)//8]
            cid=self.cid(item)
            if self.cards.db[cid]['kind']=='minion':
                self.played.add(self.cards.base[cid])
        return dec

    def __call__(self,f,c,mask,seat):
        base,v = self.original(f,c,mask,seat)
        a = int(np.argmax(base))
        if not self.enabled or not 8<=a<23:
            return base,v
        p=self.player
        sell = a>=16
        if sell and len(p['board'])!=7:
            return base,v
        offset,seq = (16,p['board']) if sell else (8,p['shop'])
        slots = [(offset+i,self.cid(item)) for i,item in enumerate(seq[:7 if sell else 8])
                 if mask[offset+i] and self.cards.db[self.cid(item)]['kind']=='minion']
        if a not in dict(slots) or len({self.cards.base[k] for _,k in slots})<2:
            return base,v
        board,hand=[self.cid(q) for q in p['board']],[self.cid(q) for q in p['hand']]
        kind='sell' if sell else 'buy_minion'
        w=self.weights[kind]['coef']
        h=np.array([self.cards.describe(k,board,hand,self.played,sell,self.current) for _,k in slots])@w
        # Remove card intercepts. Keep original probabilities and native trade type; bounded context correction.
        values=[np.log(max(float(base[i]),1e-30))+b for (i,_),b in zip(slots,np.clip(h-h.mean(),-1.5,1.5))]
        chosen=slots[int(np.argmax(values))][0]
        if chosen==a:
            return base,v
        self.adjustments[kind]+=1
        result=np.zeros_like(base)
        result[chosen]=1
        return result,v


def natural(task):
    seed,weights,out=task
    import torch
    torch.set_num_threads(1)
    field=FIELDS['transfer']
    if 'native' not in CACHE:
        CACHE['native']=(make_eval_policy(MODEL,protocol=EvalProtocol(mirror_snaps=field),device='cpu')['policy'],
                         MirrorSnapshotPool(field,sample='p50band'),Cards())
        CACHE['names']={}
    model,pool,cards=CACHE['native']
    pair=[]
    for enabled in (False,True):
        t=Trade(model,cards,weights,enabled)
        r=evaluate_policy(t,EvalConfig(games=1,seed=seed,mirror_pool=pool))
        if any(r[k] for k in ('illegal_actions','truncated_combats','truncated_games')) or (
                r['mirror']['mirror_errors'] or r['mirror']['mirror_rejected']):
            raise ValueError('invalid natural game')
        pair.append(dict(rank=r['rank_sequence'][0],eot={str(k):v for k,v in t.eot.items()},
                         adjustments=dict(t.adjustments),actions=t.actions))
    if not pair[1]['adjustments'] and pair[0]!=pair[1]:
        raise ValueError('unaffected game diverged')
    atomic_json(out/f'{seed}.json',pair)
    return pair


def main(out):
    import torch
    from hsbg.az.train.common import engine_info
    from hsbg.runtime.provenance import source_identity
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    inputs=json.loads(Path('config/research/natural12m-20261118.json').read_text())
    src=source_identity(Path.cwd())
    if src['tracked_dirty'] or engine_info()['source_hash']!=inputs['engine_fingerprint']:
        raise ValueError('source changed')
    for key,sha in ASSETS.items():
        if file_identity(DATA/key)['sha256']!=sha:
            raise ValueError('human evidence changed')
    for key,sha in inputs['asset_sha256'].items():
        if file_identity(key)['sha256']!=sha:
            raise ValueError('native asset changed')
    atomic_json(out/'run.json',dict(source=src,engine=engine_info(),human=ASSETS,assets=inputs['asset_sha256'],
        rule='Fit on human training games excluding demon-labelled games. Require lower game-balanced NLL '
             'on both separate cohorts for buy and sale versus card-only. Only then run 192 new paired full games. '
             'For substantial training require positive paired utility lower 95%, positive common growth and T12 reach >= -3pp.'))
    cards=Cards()
    weights={kind:fit(rows,len(cards.ids)) for kind,rows in examples(cards).items()}
    atomic_json(out/'fit.json',weights)
    passed=all(v['scores']['context'][cohort]['nll']<v['scores']['card_only'][cohort]['nll']
               for v in weights.values() for cohort in ('evaluation','unseen_demon'))
    if not passed:
        atomic_json(out/'complete.json',dict(human_gate=False,full_games=0))
        return
    with ProcessPoolExecutor(16) as pool:
        pairs=list(pool.map(natural,[(s,weights,out) for s in range(79510000,79510192)]))
    base,option=zip(*pairs)
    result=compare(option,base)
    changes=Counter()
    for r in option:
        changes.update(r['adjustments'])
    gate=(result['utility']['ci95'][0]>0 and result['growth']['mean']>0 and result['reach12']['mean']>=-.03)
    atomic_json(out/'complete.json',dict(human_gate=True,natural_gate=gate,full_games=384,paired=result,
               changed_games=sum(bool(r['adjustments']) for r in option),changes=dict(changes)))


if __name__=='__main__':
    main(Path(sys.argv[1]))
