"""Can public next-turn estimates pick better branches on unused future streams?"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.assess_investment_timing import UTILITY, read
from tools.compare_research_panels import bootstrap


def features(turn, previous, current):
    strength,hp,tier = current
    return [turn,hp,tier,np.log1p(strength),hp-previous[1],np.log1p(strength)-np.log1p(previous[0])]


def samples(games):
    x,y,turns = [],[],[]
    for game in games:
        row = game['parent']
        for t in range(8,14):
            board = row['eot']
            if str(t-1) in board and str(t) in board:
                x.append(features(t,board[str(t-1)],board[str(t)]))
                y.append(UTILITY[row['rank']-1])
                turns.append(t)
    return np.array(x),np.array(y),np.array(turns)


def main(dev,transfer,futures,out):
    plan = json.loads(Path('config/research/investment-timing-20261118.json').read_text())
    a,_ = read(dev,'development',plan)
    b,_ = read(transfer,'transfer',plan)
    x,y,t = samples(a)
    center,scale = x.mean(0),x.std(0)
    scale[scale<1e-6] = 1.
    z = (x-center)/scale
    weights = np.linalg.solve(z.T@z+20*np.eye(z.shape[1]),z.T@(y-y.mean()))
    def predict(x):
        return np.clip(y.mean()+((x-center)/scale)@weights,min(UTILITY),max(UTILITY))
    hx,hy,ht = samples(b)
    baseline = np.array([y[t==i].mean() for i in ht])
    complete = json.loads((futures/'complete.json').read_text())
    if complete['sources']!=36 or complete['run']!=file_identity(futures/'run.json'):
        raise ValueError('incomplete branch panel')
    cases = []
    for idx in range(36):
        r = json.loads((futures/f'{idx:02d}-returns.json').read_text())
        turn = r['event']['turn']
        def short_value(arm):
            boards = arm['eot']
            if str(turn+1) not in boards or str(turn) not in boards:
                return min(UTILITY)
            return float(predict(features(turn+1,boards[str(turn)],boards[str(turn+1)])))
        first_four = [short_value(f['results'][0])-short_value(f['results'][1]) for f in r['futures'][:4]]
        last_four = [UTILITY[f['results'][0]['rank']-1]-UTILITY[f['results'][1]['rank']-1]
                     for f in r['futures'][4:]]
        baseline_invest = r['group']!='incumbent' and r['event']['baseline']==3
        delta = float(np.mean(first_four))
        selected = baseline_invest if delta==0 else delta>0
        value = float(np.mean(last_four))
        cases.append(dict(group=r['group'],source=r['source'],short_delta=delta,
                          test_delta=value,invest=bool(selected),parent_invest=bool(baseline_invest)))
    report = {}
    for group in ('all','milestone','optional_six','incumbent'):
        cs = [r for r in cases if group=='all' or r['group']==group]
        ds = np.array([r['test_delta'] for r in cs])
        v = np.array([r['invest'] for r in cs])*ds
        base = np.array([r['parent_invest'] for r in cs])*ds
        report[group] = dict(selected_invest=sum(r['invest'] for r in cs),
                             minus_parent=bootstrap(v-base),minus_invest=bootstrap(v-ds),
                             minus_alternative=bootstrap(v))
    atomic_json(out,dict(development_games=len(a),development_rows=len(x),transfer_games=len(b),
                         transfer_rows=len(hx),turn_mean_mse=float(np.mean((baseline-hy)**2)),
                         model_mse=float(np.mean((predict(hx)-hy)**2)),groups=report,cases=cases,
                         future_panel=file_identity(futures/'complete.json')))


if __name__=='__main__':
    main(*map(Path,sys.argv[1:]))
