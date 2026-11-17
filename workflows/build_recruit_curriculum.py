"""Build paired natural/endowed curriculum starts from fresh training-only games."""
import argparse,gzip,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hsbg_engine as engine
from hsbg.az.train.common import engine_info
from hsbg.az.train.economic_tasks import own,debug,card_ids
from hsbg.az.train.recruit_start import validate_state
from hsbg.runtime.artifacts import atomic_json,file_identity

CORES=('BG_LOE_077','BG33_823','BG33_825')


def endow(source):
    game=engine.Game.from_json(json.dumps(source));edits=[];ids=card_ids()
    def edit(cmd,**kw):
        debug(game,cmd,**kw);edits.append(dict(cmd=cmd,**kw))
    core_ids={ids[x] for x in CORES+('TB_BaconUps_045','BG33_823_G','BG33_825_G')}
    for i in reversed(range(len(own(game)['board']))):
        if own(game)['board'][i]['card'] in core_ids:edit('remove',zone='board',index=i)
    while len(own(game)['board'])>4:
        board=own(game)['board'];i=min(range(len(board)),key=lambda j:board[j]['atk']+board[j]['hp'])
        edit('remove',zone='board',index=i)
    for card in CORES:edit('add_minion',zone='board',card=card)
    edit('clear_zone',zone='hand')
    for card in ('BG33_814','BG33_815'):edit('add_spell',zone='hand',card=card)
    if len(own(game)['shop'])>=8:edit('remove',zone='shop',index=7)
    edit('add_minion',zone='shop',card='BG33_821')
    state=json.loads(game.state_json());validate_state(state)
    if any(game.diagnostic_counts()):raise ValueError('unclean endowed state')
    if not {ids[x] for x in CORES}<={m['card'] for m in own(game)['board']}:
        raise ValueError('economic core setup failed')
    for key in ('gold','health','tier'):
        if own(game)[key]!=source['players'][0][key]:raise ValueError('unexpected resource/health intervention')
    return state,edits


def build(bank,out):
    bank=Path(bank);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((bank/'manifest.json').read_text())
    if file_identity(bank/'states.jsonl.gz')!=manifest['data']:raise ValueError('source identity differs')
    groups={}
    with gzip.open(bank/'states.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);p=row['state']['players'][0]
            if row['seat']!=0 or row['seed']%5==0 or not row['state']['lobby_races'] & (1<<6):continue
            if not 8<=row['turn']<=9 or p['health']<15:continue
            validate_state(row['state'])
            if row['seed'] not in groups or row['turn']<groups[row['seed']]['turn']:groups[row['seed']]=row
    if len(groups)<10:raise ValueError('too few independent source games')
    for variant in ('natural','endowed'):
        dest=out/variant;dest.mkdir()
        with gzip.open(dest/'states.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as stream:
            for seed,row in sorted(groups.items()):
                state,edits=(row['state'],[]) if variant=='natural' else endow(row['state'])
                stream.write(json.dumps({'source_seed':seed,'source_turn':row['turn'],'split':'train',
                    'state':state,'interventions':edits},separators=(',',':'))+'\n')
        atomic_json(dest/'manifest.json',{'schema':'hsai.recruit-start-bank/v1','variant':variant,
            'engine':engine_info(),'states':len(groups),'source_seeds':sorted(groups),
            'data':file_identity(dest/'states.jsonl.gz'),'source_manifest':file_identity(bank/'manifest.json'),
            'source_training_field':'artifacts/libraries/arena-cal-v14-snap',
            'scope':'Whole-state episode starts; endowment is explicit curriculum, not natural-game evidence',
            'interventions':'endowed: three normal economic cores, two bounties and one Rascal offer; gold/health/tier unchanged'})
    print(json.dumps({'source_games':len(groups),'variants':['natural','endowed']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();build(a.bank,a.out)
