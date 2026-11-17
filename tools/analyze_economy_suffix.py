"""Merge sealed conditional suffix studies and describe their EOT boards.

Read-only CPU analysis. Composition summaries describe survivor cohorts;
they do not establish why one policy wins or loses.
"""
import argparse,json,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.runtime.artifacts import atomic_json,file_identity
from workflows.probe_economy_suffix import compare


def merge_studies(roots):
    rows=[];identities=[];reference=None;models={}
    for root in map(Path,roots):
        complete=json.loads((root/'complete.json').read_text())
        run=json.loads((root/'run.json').read_text())
        if not complete.get('valid') or complete.get('probe'):raise ValueError('study not complete ability diagnostic')
        if complete['run']!=file_identity(root/'run.json'):raise ValueError('run identity differs')
        comparable={k:run[k] for k in ('engine','action_contract','source_manifest','snapshot_files')}
        if reference is not None and comparable!=reference:raise ValueError('study field differs')
        reference=comparable
        if set(models)&set(complete['models']):raise ValueError('duplicate policy name')
        models.update(complete['models'])
        if complete['rollouts']!=len(complete['rows']):raise ValueError('rollout count differs')
        for row in complete['rows']:
            if row['trace']['identity']['checkpoint_sha256']!=complete['models'][row['name']]['sha256']:
                raise ValueError('row model identity differs')
            rows.append(row)
        identities.append({'directory':str(root),'complete':file_identity(root/'complete.json'),'run':complete['run']})
    # compare verifies all arms, paired seeds, and exact initial engine state.
    comparison=compare(rows,'parent')
    return rows,identities,comparison


def board_stats(board):
    body=[int(m['atk'])+int(m['hp']) for m in board];total=sum(body)
    return {'board_n':len(board),'body':total,'largest_body_share':max(body,default=0)/max(1,total),
        'weakest_body':min(body,default=0),
        'shield_n':sum(bool(m['kw']&2) for m in board),
        'venomous_n':sum(bool(m['kw']&16) for m in board),
        'poisonous_n':sum(bool(m['kw']&32) for m in board),
        'reborn_n':sum(bool(m['kw']&8) for m in board),
        'cleave_n':sum(bool(m['kw']&512) for m in board),
        'pirate_n':sum(bool(m['races']&64) for m in board)}


def analyze(roots):
    rows,identities,comparison=merge_studies(roots)
    grouped=defaultdict(list);cards=defaultdict(Counter);sources=defaultdict(set)
    for root in map(Path,roots):
        complete=json.loads((root/'complete.json').read_text())
        for expected in complete['rows']:
            path=root/'trials'/expected['name']/f"case-{expected['case']:03d}-{expected['variant']}-r{expected['replica']:02d}"/'result.json'
            result=json.loads(path.read_text())
            if result['summary']!=expected:raise ValueError('trial summary differs')
            for t in result['evaluation']['end_turn_economy']:
                if t['eot_board'] is None:raise ValueError('missing EOT board')
                key=(expected['name'],expected['variant'],int(t['turn']))
                stats=board_stats(t['eot_board'])
                if stats['body']!=t['eot_attack']+t['eot_board_hp']:raise ValueError('EOT sum differs')
                stats.update(tier=t['tier'],tier_six=int(t['tier']==6),health=t['health'],hand_n=t['hand_n'])
                grouped[key].append(stats);sources[key].add(expected['case'])
                cards[key].update(m['card_id'] for m in t['eot_board'])
    boards={}
    for key,items in sorted(grouped.items()):
        boards['/'.join(map(str,key))]={'n':len(items),'sources':len(sources[key]),
            'means':{k:float(np.mean([r[k] for r in items])) for k in items[0]},
            'medians':{k:float(np.median([r[k] for r in items])) for k in items[0]},
            'card_occurrences':dict(cards[key].most_common())}
    return {'inputs':identities,'comparison':comparison,'boards':boards,
        'scope':'Descriptive matched endowed suffixes, not natural performance or causal attribution of board features'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',required=True);args=p.parse_args()
    if Path(args.out).exists():raise FileExistsError(args.out)
    result=analyze(args.runs);atomic_json(args.out,result)
    print(json.dumps(result['comparison']['paired']),flush=True)
