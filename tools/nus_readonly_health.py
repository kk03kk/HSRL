"""NUS登录节点有界只读审计：纯标准库；不加载模型、不写文件、不导入项目。"""
import argparse, collections, datetime, json, math, os, re, subprocess, time
from pathlib import Path

M=1024*1024
def emit(label,value):
 print(label,json.dumps(value,ensure_ascii=False,default=str,allow_nan=True))
def command(argv):
 try:
  r=subprocess.run(argv,capture_output=True,text=True,timeout=15)
  return (r.stdout+r.stderr).strip()
 except (OSError,subprocess.TimeoutExpired) as e:return str(e)
def read_json(path,limit=2*M):
 try:
  with path.open('rb') as f:data=f.read(limit+1)
  if len(data)>limit:return {'_skipped':'size_limit','bytes':path.stat().st_size}
  return json.loads(data)
 except (OSError,ValueError) as e:return {'_unavailable':str(e)}
def stat(path):
 try:
  s=path.stat();return dict(bytes=s.st_size,age_s=round(time.time()-s.st_mtime,1))
 except OSError as e:return dict(unavailable=str(e))
def tail(path,limit):
 try:
  with path.open('rb') as f:
   f.seek(0,2);f.seek(max(0,f.tell()-limit));return f.read(limit).decode('utf8','replace')
 except OSError as e:return str(e)
def select(d,keys):return {k:d[k] for k in keys.split() if k in d} if isinstance(d,dict) else d

def history(path):
 """小于16MiB读完整文件，否则首/中/尾各4MiB；内存只保留统计和最后五行。"""
 try:
  with path.open('rb') as f:
   f.seek(0,2);size=f.tell();w=4*M
   windows=[(0,size)] if size<=16*M else [(0,w),((size-w)//2,w),(max(0,size-w),w)]
   seen=set();tot=collections.Counter();bad=[];recent=collections.deque(maxlen=5);ranges=[]
   for start,length in windows:
    f.seek(start);raw=f.read(length);lines=raw.split(b'\n')
    if start:lines=lines[1:]
    partial=bool(lines[-1]);lines=lines[:-1];ids=[];metrics=collections.Counter();metric_counts=collections.Counter()
    for line in lines:
     if not line.strip():continue
     try:r=json.loads(line)
     except ValueError:
      if len(bad)<10:bad.append('complete_line_invalid_json')
      continue
     if not isinstance(r,dict) or r.get('iter') is None:continue
     i=r['iter']
     if i in seen:continue
     seen.add(i);ids.append(i);q=r.get('ppo') or {};c=r.get('collect') or {};p=r.get('profile') or {};pipe=r.get('pipeline') or {}
     tot['rows']+=1;tot['samples']+=q.get('samples',0);tot['processed']+=q.get('processed_samples',0);tot['optimizer_steps']+=q.get('updates',0)
     tot['decisions']+=c.get('decisions',0)
     tot['rows_missing_quality_fields']+=any(k not in c for k in ('engine_quality_checked_games','mirror_strict_counts'))
     tot['zero_updates']+=not bool(q.get('updates'));tot['kl_stops']+=bool(q.get('stopped_early'));tot['fallbacks']+=bool(pipe.get('fallback_zero_step'))
     for k in ('illegal_actions','truncated_games','mirror_errors','mirror_missing','mirror_fallback','mirror_attempt','mirror_applied','engine_quality_checked_games','games','truncated_combats','truncated_recruit_contexts','turn_limit_end','engine_stopped','mirror_pool_unbalanced','mirror_pool_species_unbalanced','forced_end_turns'):tot[k]+=c.get(k,0)
     for k,v in (c.get('mirror_rejected_by_reason') or {}).items():tot['reject:'+k]+=v
     tot['eligible_live_seats']+=(c.get('mirror_strict_counts') or {}).get('eligible_live_seats',0)
     for k in ('collect_s','ppo_s','parent_read_s','build_batch_s','checkpoint_save_publish_s'):tot[k]+=p.get(k,0)
     t=p.get('timeline') or {}
     if 't_history_done' in t and 't_iter_start' in t:tot['iteration_s']+=t['t_history_done']-t['t_iter_start']
     epochs=q.get('epochs') or [];e=epochs[-1] if epochs else {}
     for k in ('total','policy_loss','value_loss','approx_kl','entropy','clip_frac','ratio_mean'):
      if isinstance(e.get(k),(float,int)) and not math.isfinite(e[k]) and len(bad)<10:bad.append({'iter':i,'nonfinite':k})
      if isinstance(e.get(k),(float,int)) and math.isfinite(e[k]):metrics[k]+=e[k];metric_counts[k]+=1
     recent.append(dict(iter=i,ppo=select(q,'samples processed_samples updates lr anchor_coef stopped_early seconds'),epoch=select(e,'value_loss approx_kl entropy clip_frac ratio_mean illegal_prob_max'),pipeline=pipe,profile=select(p,'collect_s ppo_s checkpoint_save_publish_s'),config=select(q.get('config',{}),'lr target_kl epochs minibatch gamma lam value_loss entropy_coef device amp')))
    ranges.append(dict(byte_offset=start,bytes_read=len(raw),first_iter=min(ids) if ids else None,last_iter=max(ids) if ids else None,partial_last_line_ignored=partial,epoch_metric_row_means={k:round(v/metric_counts[k],6) for k,v in metrics.items()}))
   return dict(file_bytes=size,scope='whole_file' if len(windows)==1 else 'first_middle_last_windows_not_full_totals',ranges=ranges,totals=dict(tot),processed_fraction=tot['processed']/max(1,tot['samples']),effective_samples_per_iteration_s=tot['processed']/tot['iteration_s'] if tot['iteration_s'] else None,issues=bad,last_five=list(recent))
 except OSError as e:return dict(unavailable=str(e))

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,default=Path.cwd());a.add_argument('--run',default='personal/runs/nus-4159-economy-long-002');a.add_argument('--job',default='auto');args=a.parse_args()
 root=args.root.resolve();run=(root/args.run).resolve()
 if not run.is_relative_to(root):a.error('run必须在项目内')
 emit('TIME',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=root,run=run))
 queue=command(['squeue','-u',os.environ.get('USER',''),'--noheader','--format=%i|%j|%T|%M|%N']);emit('QUEUE',queue)
 fields=[list(map(str.strip,x.split('|'))) for x in queue.splitlines()]
 matches=[x[0] for x in fields if len(x)>=3 and x[1]=='hsrl-4159-long' and x[2]=='RUNNING']
 job=matches[0] if args.job=='auto' and len(matches)==1 else args.job
 if job!='auto':
  emit('SACCT',command(['sacct','-j',job,'--format=JobID,State,Elapsed,ExitCode,AllocCPUS,ReqMem,MaxRSS','-P']))
  info=command(['scontrol','show','job',job,'-o']);emit('JOB',info)
  for name in ('StdErr','StdOut'):
   m=re.search(r'(?:^|\s)'+name+r'=(\S+)',info)
   if m:emit(name+'_TAIL',tail(Path(m.group(1)),16000 if name=='StdErr' else 8000))
 else:emit('JOB_SELECTION','未唯一识别运行中的hsrl-4159-long；请回传QUEUE，不猜测其他作业')
 if not run.is_dir():
  emit('RUN_MISSING',[(str(p),stat(p/'train/latest.json')) for p in (root/'personal/runs').glob('nus-4159-economy-long-*')]);return
 plan=read_json(run/'plan.json');emit('PLAN',plan);emit('CONFIG',read_json(run/'train.config.json'))
 for name in ('status.json','learning-health.json','FAILED.json','engineering-link.json'):
  p=run/name;emit(name,dict(stat=stat(p),data=read_json(p)))
 before=read_json(run/'train/latest.json');emit('LATEST_BEFORE',dict(stat=stat(run/'train/latest.json'),data=before))
 base=(plan.get('inherited') or {}).get('actual_decisions');cum=before.get('cum_decisions')
 emit('PROGRESS',dict(inherited_decisions=base,cumulative_decisions=cum,new_decisions=cum-base if isinstance(base,int) and isinstance(cum,int) else None))
 for name in ('model','trainer'):
  if isinstance(before.get(name),str):emit('LATEST_'+name.upper()+'_STAT',stat(Path(before[name])))
 emit('HISTORY',dict(stat=stat(run/'train/history.jsonl'),data=history(run/'train/history.jsonl')))
 report=read_json(run/'training-report.json');emit('CACHED_TRAIN_REPORT',dict(stat=stat(run/'training-report.json'),data=select(report,'totals rates mirror_strict_counts mirror_rejected_by_reason unexpected_mirror_rejections missing_timeline_iterations')))
 def diagnostics(path,label):
  d=read_json(path);parts={}
  for key in ('reference','candidate'):
   q=d.get(key,{}) if 'reference' in d else d if key=='reference' else {}
   parts[key]=select(q,'checkpoint_sha256 protocol_sha256 mean_rank mean_rank_ci95 top4_rate win_rate economy_stage')
   parts[key]['turns']={str(t):select((q.get('turns') or {}).get(str(t),{}),'reached_end_turn_rate reached_end_turn mean_tier mean_strength mean_gold_left mean_actions mean_actions_by_type upgrade_taken_per_available_turn live_win_rate mean_damage_taken forced_end_turns') for t in (3,6,8,10,12,14,15,17)}
  emit(label,dict(path=path,stat=stat(path),summary=parts))
 diagnostics(run/'baselines/diagnostics.json','BASELINE')
 milestones=sorted((run/'milestones').glob('decisions-*'))[-40:]
 for m in milestones:
  d=read_json(m/'report.json');emit('MILESTONE',dict(path=m,report=select(d,'target checkpoint paired monitor_skipped stopped_for_performance')))
  emit('MILESTONE_FILES',dict(path=m,model=stat(m/'model.pt'),trainer=stat(m/'trainer.pt')))
 for m in milestones[-2:]:diagnostics(m/'diagnostics.json','RECENT_DIAGNOSTICS')
 bank=root/'personal/artifacts/libraries/bot-firestone-3702-economy-v2-live-v1'
 bank_report=read_json(bank/'report.json');emit('BANK_REPORT',select(bank_report,'complete fresh_generation smoke fingerprint source_games coverage calibration smoke_validation'))
 for split in ('train','dev','test'):
  d=read_json(bank/split/'manifest.json');emit('BANK_'+split,select(d,'schema split digest fingerprint games validation'))
 emit('LATEST_AFTER',read_json(run/'train/latest.json'))
 emit('STOP_FILE_EXISTS',(run/'STOP_REQUESTED').exists())
 emit('GIT_HEAD',command(['git','-C',str(root),'rev-parse','HEAD']))
 emit('DISK',command(['df','-h',str(run)]))
 emit('NOTE','在线只读快照；latest变化和末行未写完正常。缓存汇总可能滞后。未查模型内容、未读取Test对局，未写任何项目文件。')
if __name__=='__main__':main()
