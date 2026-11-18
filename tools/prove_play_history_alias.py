"""Construct two native histories with identical model inputs and different teacher memory.

This establishes teacher-representability limits, not optimal-action ambiguity
or frequency in natural play. All setup interventions are explicit.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.economic_tasks import debug
from hsbg.az.train.research_economy import Operator
from hsbg.az.train import mechanism_obs as M
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity


def public(game):
    v=json.loads(game.view(0))
    return {k:v[k] for k in ('me','mask','turn','phase','lobby_races')}


def construct(source):
    import hsbg_engine as E
    from workflows.probe_whole_turn_economy import fixture
    state,edits=fixture(source,'complete_gold');game=E.Game.from_json(json.dumps(state))
    def edit(cmd,**kw):
        debug(game,cmd,**kw);edits.append({'cmd':cmd,**kw})
    for zone in ('board','hand','shop'):edit('clear_zone',zone=zone)
    edit('set_player',tier=6,gold=10)
    for card in ('TB_BaconUps_045','BG33_825_G','BG33_823','BG36_523','AV_113t3t2','BG22_HERO_001p_t1et'):
        edit('add_minion',zone='board',card=card)
    for card in ('AV_113t3t2','BG22_HERO_001p_t1et'):edit('add_minion',zone='hand',card=card)
    initial=json.loads(game.state_json());histories=[];observations=[]
    # Both play A and B once, sell A and B once, and end with the same ordered
    # visible board. They differ only in which old/new entities were sold.
    for label,actions in (('retain_new_A',(50,21,52,22)),('retain_new_B',(50,20,52,21))):
        game=E.Game.from_json(json.dumps(initial));operator=Operator('growth');steps=[]
        for action in actions:
            before=public(game);advice,reason=operator.act(before)
            if not bytes(game.legal_mask(0))[action]:raise ValueError('constructed action is not legal')
            native_before=json.loads(game.state_json());game.step(0,action);after=public(game)
            if game.turn!=initial['turn'] or not after['me']['alive'] or any(game.diagnostic_counts()):
                raise ValueError('constructed native path is not clean')
            operator.note(before,after,action)
            steps.append({'action':action,'teacher_advice_not_executed':advice,'teacher_reason':reason,
                'before':native_before,'after':json.loads(game.state_json())})
        view=public(game);advice,reason=operator.act(view)
        legacy,cards=game.observe(0);features,cards_m=M.observe(game,0,M.SCHEMA);mask=bytes(game.legal_mask(0))
        if bytes(cards)!=bytes(cards_m):raise ValueError('observation card layout differs')
        observations.append((bytes(legacy),bytes(features),bytes(cards),mask))
        histories.append({'name':label,'steps':steps,'final':json.loads(game.state_json()),'public_final':view,
            'cycled_eids':sorted(operator.cycled),'cycled_board_slots':[i for i,m in enumerate(view['me']['board']) if m['eid'] in operator.cycled],
            'teacher_action':advice,'teacher_reason':reason})
    names=('legacy_features','mechanism_features','cards','legal_mask')
    equal={k:observations[0][i]==observations[1][i] for i,k in enumerate(names)}
    if not all(equal.values()):raise ValueError('constructed native histories are distinguishable: '+str(equal))
    if histories[0]['teacher_action']==histories[1]['teacher_action']:
        raise ValueError('teacher actions did not separate')
    hashes={k:hashlib.sha256(observations[0][i]).hexdigest() for i,k in enumerate(names)}
    return {'passed':True,'initial':initial,'setup_interventions':edits,'histories':histories,
        'input_equal':equal,'input_sha256':hashes,'mechanism_schema':M.SCHEMA,'mechanism_schema_sha256':M.SCHEMA_SHA256,
        'claim':'Current deterministic feedforward policies receive identical inputs on these two native action histories, but cannot reproduce both history-dependent teacher actions',
        'limitations':['Explicitly constructed token/minion setup, not observed natural frequency',
            'Does not prove either teacher sale is optimal or that the optimal game policy requires memory',
            'The histories share public cards and encoded stats, not native entity IDs or full simulator state',
            'Only an input/teacher representability certificate, not a model ability benchmark']}


def main(args):
    import hsbg_engine as E
    game=E.Game(seed=11800000)
    while game.turn<8 and not game.is_over():
        for seat in range(game.num_players):
            if game.needs_action(seat):game.bot_play(seat)
    while json.loads(game.player_json(0))['pending']:game.step(0,game.bot_action(0))
    source=json.loads(game.state_json());result=construct(source)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    with gzip.open(out/'native-proof.json.gz','wt',encoding='utf-8') as stream:
        stream.write(json.dumps({'source':source,**result},separators=(',',':'))+'\n')
    result={k:v for k,v in result.items() if k not in ('initial','histories','setup_interventions')}
    result.update(source=source_identity(Path(__file__).resolve().parents[1]),proof=file_identity(out/'native-proof.json.gz'))
    atomic_json(out/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);main(p.parse_args())
