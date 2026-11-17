"""Descriptive paired analysis; never changes the protocol or promotes a model."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.az.train.eval_policy import assert_same_protocol
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.compare_panels import per_turn


def paired_mean_delta(before, after, *, rounds=10000):
    delta = np.asarray(after, dtype=float)-np.asarray(before, dtype=float)
    if not len(delta): return {'n':0}
    rng = np.random.default_rng(20261117)
    samples = delta[rng.integers(0, len(delta), size=(rounds,len(delta)))].mean(axis=1)
    return {'n':len(delta), 'delta':float(delta.mean()),
            'ci95':np.quantile(samples,[.025,.975]).tolist()}


def analyze(panel_path, baseline, candidate, out):
    panel_path, out = Path(panel_path), Path(out)
    panel = json.loads(panel_path.read_text(encoding='utf-8'))
    protocol = assert_same_protocol(panel)
    complete = json.loads((panel_path.parent/'complete.json').read_text(encoding='utf-8'))
    if complete.get('valid') is not True or complete['panel'] != file_identity(panel_path):
        raise ValueError('panel has no matching successful completion record')
    models = {m['name']:m for m in panel['models']}
    a,b = models[baseline],models[candidate]
    if a['seed_sequence'] != b['seed_sequence']:
        raise ValueError('analysis requires identical ordered seeds')
    ra,rb = np.array(a['rank_sequence']),np.array(b['rank_sequence'])
    if not np.all((ra>=1)&(ra<=8)&(rb>=1)&(rb<=8)): raise ValueError('unfinished ranks')
    rank = paired_mean_delta(ra,rb)
    rank.update(candidate_better=int((rb<ra).sum()), baseline_better=int((ra<rb).sum()), ties=int((ra==rb).sum()))
    turns = sorted({int(t) for m in (a,b) for t in m['curves']})
    result = {'schema':'hsai.paired-analysis/v1', 'protocol_sha256':protocol,
              'panel':file_identity(panel_path), 'baseline':baseline, 'candidate':candidate,
              'interpretation':'delta=candidate-baseline; rank negative is better; seed-paired separate lobbies, not head-to-head matches; pointwise exploratory intervals, no multiple-testing claim',
              'rank':rank, 'top4':paired_mean_delta((ra<=4).astype(float),(rb<=4).astype(float)),
              'win':paired_mean_delta((ra==1).astype(float),(rb==1).astype(float)),
              'per_turn':per_turn(models,baseline,candidate,turns), 'growth':[], 'models':{}}
    for start,end in ((1,5),(5,10),(10,12),(12,16)):
        sa,sb = a['strength_by_seed'],b['strength_by_seed']
        common = sorted(s for s in set(sa)&set(sb) if all(str(t) in m[s] for m in (sa,sb) for t in (start,end)))
        ga=[sa[s][str(end)]-sa[s][str(start)] for s in common]
        gb=[sb[s][str(end)]-sb[s][str(start)] for s in common]
        result['growth'].append({'from_turn':start,'to_turn':end,'n':len(common),
                                  'baseline_gain_p50':float(np.median(ga)) if ga else None,
                                  'candidate_gain_p50':float(np.median(gb)) if gb else None,
                                  'mean_gain_delta':paired_mean_delta(ga,gb)})
    for name,m in ((baseline,a),(candidate,b)):
        rows = m['trace']
        phase_stats = {}
        for label,lo,hi in (('early',1,5),('middle',6,10),('late',11,100)):
            games=defaultdict(list)
            for row in rows:
                if lo<=row['turn']<=hi: games[row['seed']].append(row)
            def per_game_mean(key):
                values = [np.mean([r[key] for r in rs if r.get(key) is not None])
                          for rs in games.values() if any(r.get(key) is not None for r in rs)]
                return float(np.mean(values)) if values else None
            phase_stats[label]={'turn_range':[lo,hi], 'games':len(games), 'turn_rows':sum(map(len,games.values())),
                                'gold_left_mean_per_game':per_game_mean('gold'),
                                'actions_mean_per_game':per_game_mean('actions'),
                                'hand_left_mean_per_game':per_game_mean('hand_n')}
        tier_arrivals={}
        for tier in range(2,7):
            first={}
            for row in rows:
                if row['tier']>=tier: first[row['seed']]=min(first.get(row['seed'],999),row['turn'])
            tier_arrivals[str(tier)]={'games':len(first),'share':len(first)/len(m['rank_sequence']),
                                      'first_eot_turn_p50':float(np.median(list(first.values()))) if first else None}
        result['models'][name]={k:m[k] for k in ('outcome','survival','action_hist','field_stats','curves',
                                               'action_mix_by_turn','mechanics_by_turn','gold_by_turn','retained_gain_by_turn')}
        result['models'][name].update(phases=phase_stats,tier_arrivals=tier_arrivals,
                                      decision_trace_bytes=sum(r['bytes'] for r in m['decision_traces']))
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    atomic_json(out/'analysis.json',result)
    plot(result,out/'growth.png')
    return result


def plot(result, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name in (result['baseline'],result['candidate']):
        rows=result['per_turn']
        axes[0,0].plot([r['turn'] for r in rows],[r[name]['m1'] for r in rows],marker='.',label=name)
        paired=[r for r in rows if r.get('paired')]
        axes[0,1].plot([r['turn'] for r in paired],[r['paired'][name] for r in paired],marker='.',label=name)
        axes[1,0].plot([r['turn'] for r in rows],[r[name]['n'] for r in rows],marker='.',label=name)
        axes[1,1].plot([r['turn'] for r in rows],[r[name]['m2'] for r in rows],marker='.',label=name)
    for ax,title in zip(axes.flat,('Conditional board strength P50','Common-survivor board strength P50',
                                 'Surviving games at each turn','Mean tavern tier')):
        ax.set_title(title);ax.set_xlabel('Turn');ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Same protocol, same seeds; late-turn samples are selected survivors')
    fig.savefig(destination,dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel',type=Path)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.panel,args.baseline,args.candidate,args.out)
    print(json.dumps({k:result[k] for k in ('rank','top4','win','growth')}))
