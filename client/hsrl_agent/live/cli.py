"""Offline default. Native I/O is imported only for explicit run/calibrate."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from .contracts import Blocked
from .runtime import load,save,catalog


def parser(root):
    p=argparse.ArgumentParser(description='Pinned BG policy bridge (offline compatibility required); run is shadow unless --enable-input')
    sub=p.add_subparsers(dest='command',required=True)
    from .selection import selected_checkpoint
    default=str(selected_checkpoint(root))
    q=sub.add_parser('doctor');q.add_argument('--checkpoint',default=default);q.add_argument('--infer',action='store_true')
    q.add_argument('--profile',default=str(root/'config/live/autoplay-255790.json'));q.add_argument('--game-dir')
    q.add_argument('--out',default=str(root/'artifacts/runs/live-doctor.json'))
    q=sub.add_parser('replay');q.add_argument('--power-log',required=True);q.add_argument('--scene-log')
    q.add_argument('--limit',type=int,default=0);q.add_argument('--out',required=True);q.add_argument('--facts')
    q.add_argument('--races',nargs=5);q.add_argument('--player-id',type=int)
    q=sub.add_parser('decide');q.add_argument('--snapshot',required=True);q.add_argument('--checkpoint',default=default)
    q.add_argument('--out',required=True);q.add_argument('--device',default='cpu')
    for command in ('run','calibrate','autoplay'):
        q=sub.add_parser(command);q.add_argument('--game-dir',required=True)
        q.add_argument('--profile',required=command!='autoplay',default=str(root/'config/live/autoplay-255790.json'));q.add_argument('--out',required=True);q.add_argument('--player-id',type=int)
        if command in ('run','autoplay'):
            q.add_argument('--checkpoint',default=default);q.add_argument('--device',default='cpu')
            q.add_argument('--facts');q.add_argument('--enable-input',action='store_true');q.add_argument('--seconds',type=float,default=300)
            q.add_argument('--timing',choices=('balanced','conservative'),default='balanced')
            q.add_argument('--inference-threads',type=int,default=8,help='bounded CPU inference parallelism, 1..32')
            if command=='autoplay':
                q.set_defaults(seconds=3600);q.add_argument('--shadow',action='store_true');q.add_argument('--no-queue',action='store_true')
                q.add_argument('--games',type=int,default=1,help='games to run; zero runs until STOP/F8')
                q.add_argument('--no-auto-focus',action='store_true',help='leave window activation to the operator')
                q.add_argument('--queue-retries',type=int,default=3,help='maximum verified failed-match restarts')
                q.add_argument('--queue-timeout',type=float,default=240,help='total bounded lobby/matchmaking seconds')
                q.add_argument('--max-failures',type=int,default=3,help='failed matches a batch may absorb')
                q.add_argument('--max-same-failure',type=int,default=2,help='repeats of one incident signature before the batch stops')
        else:q.add_argument('--scene',choices=('hero','recruit','discover'),required=True)
    q=sub.add_parser('review-profile');q.add_argument('--profile',required=True)
    q.add_argument('--overlay-reviewed',action='store_true',required=True)
    q=sub.add_parser('readiness');q.add_argument('--profile');q.add_argument('--coverage');q.add_argument('--archive-report');q.add_argument('--out',required=True)
    q=sub.add_parser('archive-frame');q.add_argument('--image',required=True);q.add_argument('--metadata',required=True)
    q.add_argument('--profile',default=str(root/'config/live/reference-2560x1600.json'));q.add_argument('--out',required=True)
    return p


def main(argv=None,root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[3]
    args=parser(root).parse_args(argv)
    try:
        if args.command in ('run','autoplay'):
            if not 1<=args.inference_threads<=32:raise Blocked('inference threads must be 1..32')
            from .selection import validate_live_profile
            validate_live_profile(args.checkpoint,load(args.profile))
        if args.command=='readiness':
            from .readiness import report
            r=report(load(args.profile) if args.profile else None,load(args.coverage) if args.coverage else None,
                     load(args.archive_report) if args.archive_report else None)
            save(args.out,r);print(json.dumps(r,ensure_ascii=False,indent=2));return 0
        if args.command=='archive-frame':
            from .archive import ArchivedFrame,overlay
            out=Path(args.out).resolve()
            if not out.is_relative_to(root.resolve()):raise Blocked('archive output must be inside main project')
            out.mkdir(parents=True,exist_ok=False)
            frame=ArchivedFrame.load(args.image,args.metadata);p=load(args.profile)
            p.update(review_state='historical_only',historical_source=frame.evidence())
            result=overlay(frame,p,out/'overlay.png')
            save(out/'evidence.json',dict(frame=frame.evidence(),geometry=result,current_client_authorized=False))
            save(out/'profile.json',p);print('Historical overlay saved; current client not authorized:',out);return 0
        if args.command=='doctor':
            from .policy import POLICY_SPECS
            from hsrl_agent.compat import verify
            report=dict(contract=verify(),checkpoint=args.checkpoint,
                        sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),native_input_acquired=False)
            report['checkpoint_ok']=report['sha256'] in POLICY_SPECS
            report['policy_id']=POLICY_SPECS.get(report['sha256'],{}).get('id')
            report['live_ready']=False
            report['live_blockers']=['current runtime still requires user-executed real-game acceptance']
            report['input_adapter']='public-approx-v1; explicitly lossy'
            from .selection import validate_live_profile
            import importlib.util
            report['dependencies']={name:importlib.util.find_spec(name) is not None for name in ('rapidocr','onnxruntime','PIL')}
            profile=load(args.profile)
            try:
                validate_live_profile(args.checkpoint,profile)
                report['model_profile_compatible']=True
            except Blocked as exc:
                report['model_profile_compatible']=False
                report['live_blockers'].append(str(exc))
            from .practical import PracticalGeometry
            PracticalGeometry(profile)
            report['profile_ok']=profile.get('review_state')=='reviewed' and profile.get('client_size')==[2560,1600] and profile.get('dpi')==144
            report['installation_checked']=bool(args.game_dir)
            if args.game_dir:
                exe=Path(args.game_dir)/'Hearthstone.exe'
                report['executable_matches']=exe.is_file() and hashlib.sha256(exe.read_bytes()).hexdigest()==profile.get('executable_sha256')
            if args.infer:
                from .policy import ModelPolicy
                policy=ModelPolicy(args.checkpoint)
                report['parameters']=sum(p.numel() for p in policy.model.parameters())
                import torch
                from hsbg.az.train.ppo import forward_model,split_model_outputs
                mask=torch.zeros((1,245),dtype=torch.bool);mask[:,0]=True
                with torch.inference_mode():
                    logits,_,_=split_model_outputs(forward_model(policy.model,torch.zeros((1,3227)),torch.zeros((1,124),dtype=torch.long),mask))
                report['synthetic_forward_ok']=list(logits.shape)==[1,245] and bool(torch.isfinite(logits).all())
            report['ok']=report['contract']['ok'] and report['checkpoint_ok']
            report['ready_for_user_test']=bool(report['ok'] and report['profile_ok'] and report['model_profile_compatible'] and all(report['dependencies'].values())
                and report.get('executable_matches') and report.get('synthetic_forward_ok'))
            save(args.out,report);print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['ok'] else 2
        if args.command=='replay':
            from .logs import LogReader
            from .projection import Projector
            from .facts import context_facts
            reader=LogReader(root,args.player_id);path=Path(args.power_log)
            if args.scene_log:
                for line in Path(args.scene_log).read_text(encoding='utf-8-sig').splitlines():reader.feed(line,'scene',path.stat().st_mtime)
            n=0
            with path.open(encoding='utf-8-sig') as f:
                for n,line in enumerate(f,1):
                    reader.feed(line,'power',path.stat().st_mtime)
                    if args.limit and n>=args.limit:break
            reader.caught_up=True
            data=reader.state.observe(source_updated_at=reader.latest_time).to_dict()
            info=catalog(root);snapshot=None;reason=None
            try:
                if reader.spectating:raise Blocked('spectator session cannot authorize player input')
                facts=context_facts(reader,data,info,lobby_races=args.races,supplied=load(args.facts) if args.facts else None)
                profile=dict(allowed_builds=[data['build']],max_source_age=15)
                # Replay clock is explicitly the historical log clock. The live
                # runner has no override for freshness checks.
                snapshot=Projector(profile,info).build(reader,data,now=reader.latest_time,facts=facts)
                from .encoding import encode
                import hsbg_engine as t
                encode(snapshot['view'],{t.card_info(i)[0]:i for i in range(1,t.N_CARDS)})
            except (Blocked,ValueError,KeyError,TypeError) as exc:reason=str(exc)
            report=dict(lines=n,client=data,snapshot=snapshot,blocked=reason,history_complete=reader.history_complete,
                        native_input_acquired=False,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            save(args.out,report);print(json.dumps(dict(lines=n,blocked=reason,unknown=(snapshot or {}).get('view',{}).get('unknown',[])),ensure_ascii=False))
            return 2 if reason else 0
        if args.command=='decide':
            from .actions import bind_actions
            from .policy import ModelPolicy
            s=load(args.snapshot);decision=ModelPolicy(args.checkpoint,args.device).decide(s,bind_actions(s))
            save(args.out,asdict(decision));print(json.dumps(asdict(decision),ensure_ascii=False));return 0
        if args.command=='review-profile':
            p=load(args.profile)
            if not p.get('scene_guards') or not p.get('executable_sha256'):raise Blocked('capture a real calibration before review')
            p['review_state']='reviewed';save(args.profile,p);return 0
        if args.command=='calibrate':
            from .runtime import calibrate
            return calibrate(root,args)
        if args.command=='autoplay':
            from .practical import autoplay
            return autoplay(root,args)
        from .runtime import run
        return run(root,args)
    except (Blocked,FileNotFoundError,ValueError,KeyError,TypeError) as exc:
        print('BLOCKED: '+str(exc),file=sys.stderr);return 2
