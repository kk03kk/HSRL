"""在历史公开快照上演练已登记模型。没有实时窗口或输入后端。"""
from pathlib import Path
import argparse,json,zipfile,copy,sys
import gzip
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'client')]
from hsrl_agent.live.selection import selected_checkpoint,validate_local_pin
from hsrl_agent.live.policy import ModelPolicy,configure_threads
from hsrl_agent.live.actions import bind_actions
from hsrl_agent.live.pipeline import bind_unknown_cards
from hsrl_agent.live.practical import PracticalGeometry
from hsrl_agent.live.controller import Controller
from hsrl_agent.live.contracts import Binding,Blocked


def rehearse_player150(trace, policy, out, limit):
    """Reuse the public snapshots and allowed bindings captured before dispatch."""
    cases = []
    with gzip.open(trace, 'rt', encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('event') != 'decision':
                continue
            snapshot = row['snapshot']
            if snapshot.get('action_profile') != policy.action_profile:
                raise Blocked('trace has a different action profile')
            bindings = [Binding(**entry) for entry in row['bindings']]
            if not bindings or any(not 0 <= entry.action < 245 for entry in bindings):
                raise Blocked('trace contains no usable legal action mask')
            bind_unknown_cards(snapshot['view'], policy, snapshot['approximation'])
            decision = policy.decide(snapshot, bindings)
            cases.append(dict(decision_id=row['decision_id'], action=decision.action,
                              previous_action=row['decision']['action'], legal=decision.binding in bindings))
            if limit and len(cases) >= limit:
                break
    report = dict(policy_id=policy.policy_id, checkpoint_sha256=policy.sha, action_profile=policy.action_profile,
                  observation_width=policy.input_width, cases=len(cases),
                  legal_decisions=sum(row['legal'] for row in cases), native_backend_created=False,
                  input_sent=False, current_game_verified=False)
    (out / 'report.json').write_text(json.dumps(dict(report, rows=cases), ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if cases and all(row['legal'] for row in cases) else 2

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'artifacts/local/client-v1.1-rehearsal');p.add_argument('--limit',type=int,default=55)
 p.add_argument('--checkpoint',type=Path,default=selected_checkpoint(ROOT));p.add_argument('--trace',type=Path);a=p.parse_args()
 if a.out.exists():p.error('输出目录已存在；另选新目录保留旧报告')
 configure_threads(2)
 validate_local_pin(ROOT,a.checkpoint)
 policy=ModelPolicy(a.checkpoint)
 if policy.player_observation and not a.trace:p.error('Player150 演练需要 --trace 指定已记录的决策轨迹')
 a.out.mkdir(parents=True)
 if a.trace:
  return rehearse_player150(a.trace,policy,a.out,a.limit)
 geometry=PracticalGeometry(json.loads((ROOT/'config/live/autoplay-255790.json').read_text(encoding='utf8')))
 class NoInput:
  def execute(self,*args,**kwargs):raise AssertionError('offline rehearsal attempted input')
 class FixedPolicy:
  sha=policy.sha
  def decide(self,*args):return decision
 source=ROOT/'records/evidence/live/autotavern/live-game-fix-20261029.zip';rows=[]
 with zipfile.ZipFile(source) as z:
  member=next(n for n in z.namelist() if n.endswith('/before/events.jsonl'))
  for line in z.open(member):
   event=json.loads(line)
   if event.get('event')!='model_decision':continue
   snapshot=copy.deepcopy(event['snapshot']);row=dict(index=len(rows),build=snapshot.get('build'),turn=snapshot['client']['turn'])
   try:
    bind_unknown_cards(snapshot['view'],policy,snapshot.setdefault('approximation',dict(fields={})))
    bindings=bind_actions(snapshot);decision=policy.decide(snapshot,bindings)
    assert decision.binding in bindings
    row.update(model_ok=True,action=decision.action,kind=decision.binding.kind,bindings=len(bindings))
    try:
     control=Controller(FixedPolicy(),geometry,NoInput(),a.out/f"case-{len(rows):03d}",enable_input=False)
     assert control.tick(snapshot)=='dry_run';row['geometry_and_controller_ok']=True
    except Blocked as exc:row.update(geometry_and_controller_ok=False,geometry_block=str(exc))
   except Blocked as exc:row.update(model_ok=False,blocked=str(exc))
   rows.append(row)
   if len(rows)>=a.limit:break
 report=dict(checkpoint_sha256=policy.sha,policy_id=policy.policy_id,parameters=sum(p.numel() for p in policy.model.parameters()),
     source_archive=str(source.relative_to(ROOT)),historical_builds=sorted({str(r['build']) for r in rows}),cases=len(rows),
     model_success=sum(r.get('model_ok',False) for r in rows),geometry_controller_success=sum(r.get('geometry_and_controller_ok',False) for r in rows),
     native_backend_created=False,input_sent=False,current_game_verified=False,rows=rows)
 (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))
 return 0 if rows and all(r.get('model_ok') for r in rows) else 2
if __name__=='__main__':raise SystemExit(main())
