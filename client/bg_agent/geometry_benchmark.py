"""Distinct recorded frames and authored changing-state chains; never live input."""
import argparse,json,time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from .adaptive_spatial import AdaptiveSpatialMapper
from .chain_metrics import ChainMetrics
from .geometry_mapper import GeometryMapper
from .layout_resolver import LayoutCatalog,LayoutResolver,Viewport
from .layout_study import visual_projection
from .model import Action,Card,InteractionContext,Kind,Observation,Phase,Resolution
from .resources import calibration_root
from .verification import verify,Status


def changing_chain(length):
    """Authored effects, not a Hearthstone simulator or sampled game trajectory."""
    obs=Observation(build='254477',game_id='authored',mode='GT_BATTLEGROUNDS',phase=Phase.RECRUIT,
        local_player_id=1,turn=1,gold=1000,tavern_tier=1)
    rows=[];next_id=100
    for step in range(length):
        next_id+=10
        if step%5==0:
            count=2+(step//5)%3
            choices=tuple(Card(next_id+i,card_id=f'choice-{step}-{i}',controller=1,position=i+1,card_type='MINION') for i in range(count))
            before=replace(obs,turn=step//5+1,board=(),phase=Phase.CHOICE,choices=choices,interaction=InteractionContext('choice',request_id=step+1,protocol='fixture'))
            action=Action(Kind.CHOOSE,choices[0].entity_id,choice_id=step+1)
            after=replace(before,phase=Phase.RECRUIT,choices=(),interaction=None,hand=before.hand+(replace(choices[0],zone='HAND',position=len(before.hand)+1),))
        elif step%5 in (1,3):
            card=obs.hand[0];before=obs;action=Action(Kind.PLAY,card.entity_id)
            after=replace(before,hand=tuple(replace(c,position=i+1) for i,c in enumerate(before.hand[1:])),board=before.board+(replace(card,zone='PLAY',position=len(before.board)+1),))
        elif step%5==2:
            shop=tuple(Card(next_id+i,card_id=f'shop-{step}-{i}',controller=2,zone='PLAY',position=i+1,card_type='MINION') for i in range(1+step%7))
            before=replace(obs,shop=shop);action=Action(Kind.BUY,shop[0].entity_id)
            after=replace(before,gold=before.gold-3,shop=tuple(replace(c,position=i+1) for i,c in enumerate(shop[1:])),hand=(replace(shop[0],controller=1,zone='HAND',position=1),))
        else:
            before=obs;action=Action(Kind.SELL,obs.board[0].entity_id)
            after=replace(before,gold=before.gold+1,board=tuple(replace(c,position=i+1) for i,c in enumerate(before.board[1:])))
        before=replace(before,actions=(action,))
        receipt=Resolution(step+1,action.kind,action.source,action.target,origin='server_power')
        after=replace(after,revision=step+1,resolutions=obs.resolutions+(receipt,))
        rows.append((before,action,after));obs=after
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True);parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    inventory=json.loads(args.inventory.read_text(encoding='utf8'));root=args.root.resolve();calibration=calibration_root()
    def legacy():return AdaptiveSpatialMapper(calibration/'v05_ui/profile.json',calibration/'v05_hand/profile.json')
    mapper=GeometryMapper(legacy());rows=[];seen=set();failures=[]
    # Admission probes are separate from timed passes and use no full locator.
    for row in inventory['frames_inventory']:
        if row['phase']!='recruit' or row['sha256'] in seen:continue
        source=json.loads((root/row['metadata']).read_text(encoding='utf8'));sample=source.get('sample',source)
        if str(source.get('stage','')).startswith(('branch','target','choice','confirm')):continue
        obs=Observation.from_dict(sample['observation'])
        if not obs.ready:continue
        with Image.open(root/row['image']) as original:image=original.convert('RGB')
        try:
            with patch.object(mapper.legacy,'map',side_effect=AssertionError('full_locator_forbidden')):mapper.map(obs,image,sample.get('window',{}))
        except AssertionError:continue
        if mapper.used_geometry:rows.append((row,obs,image,sample.get('window',{})));seen.add(row['sha256'])
        if len(rows)>=100:break
    results=[]
    for count in (20,50,100):
        if len(rows)<count:results.append(dict(steps=count,status='insufficient_distinct_frames',available=len(rows)));continue
        timings={};counters={};old=legacy();visuals={};differences=[];compared=0
        for mode in ('legacy','geometry'):
            metrics=ChainMetrics()
            for row,obs,image,window in rows[:count]:
                with metrics.measure('coordinate_pipeline'):
                    if mode=='legacy':
                        visual=visual_projection(old,obs,image,window)
                        visuals[row['sha256']]={m.key:m.rect for m in visual if m.status=='matched' and m.rect}
                    else:
                        with patch.object(mapper.legacy,'map',side_effect=AssertionError('full_locator_forbidden')):mapper.map(obs,image,window)
                with metrics.measure('geometry_resolve'):
                    mapper.resolver.resolve(obs,Viewport.from_window(image.size,window))
                if mode=='geometry':
                    for mapping in mapper.projection.mappings:
                        measured=visuals[row['sha256']].get(mapping.key)
                        if measured is None:continue
                        compared+=1;x,y=mapping.rect.point()
                        if not (measured.x-2<=x<=measured.x+measured.width+2 and measured.y-2<=y<=measured.y+measured.height+2):
                            if mapping.key=='region:board':
                                disposition='visual_recovery_required' if not mapper.drop_ready(image,mapping.rect) else 'calibrated_empty_append_differs_from_legacy_generic_receiver'
                            else:disposition='review_required'
                            differences.append(dict(file=row['metadata'],region=mapping.key,geometry=[x,y],visual=vars(measured),disposition=disposition))
                            if disposition=='review_required':failures.append(dict(stage='coordinate_comparison',file=row['metadata'],region=mapping.key))
            timings[mode]=metrics.report()
        a=timings['legacy']['latency_ms']['coordinate_pipeline']['total_ms'];b=timings['geometry']['latency_ms']['coordinate_pipeline']['total_ms']
        results.append(dict(steps=count,distinct_real_captures=count,kind='independent_recorded_frames_not_causal_actions',timings=timings,
            compared_regions=compared,coordinate_differences=differences,coordinate_pipeline_reduction=1-b/a))
    chains=[];resolver=LayoutResolver(LayoutCatalog.load(Path(__file__).parent/'data/layout_catalog.json'))
    for count in (20,50,100):
        metrics=ChainMetrics();statuses=[];gaps={}
        for before,action,after in changing_chain(count):
            with metrics.measure('geometry_resolve'):projection=resolver.resolve(before,Viewport(1280,800))
            for missing in projection.missing:gaps[missing]=gaps.get(missing,0)+1
            with metrics.measure('result_verification'):outcome=verify(action,before,after)
            statuses.append(str(outcome.status))
            if outcome.status!=Status.CONFIRMED:failures.append(dict(step=len(statuses),length=count,status=str(outcome.status),reason=outcome.reason))
        chains.append(dict(steps=count,kind='authored_memory_effects_not_real_game',verified=statuses.count('confirmed'),
            layouts_missing=gaps,timings=metrics.report(),final_revision=after.revision,final_gold=after.gold))
    report=dict(schema=1,recorded_frame_comparison=results,changing_state_chains=chains,failures=failures,
        excluded=['real capture latency','game animations and server wait','mouse timings','initial calibration','full live operation chain'],
        full_chain_30_percent_target='pending_live_measurement',live_game_verified=False)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=True))
    return bool(failures)


if __name__=='__main__':raise SystemExit(main())
