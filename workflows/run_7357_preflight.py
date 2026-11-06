"""一次GPU工程验收：完整状态→更新→保存恢复→显式课程换库→继续更新。"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from workflows.run_7357_long import defaults,inputs,MODEL_SHA,TRAINER_SHA,read

def main():
    p=defaults(argparse.ArgumentParser(description=__doc__));p.add_argument('--out',type=Path,default='artifacts/runs/p50-7357-preflight');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('GPU计算节点专用')
    if a.out.exists() and any(not f.name.startswith('visibility-') for f in a.out.iterdir()):p.error('预检输出必须新目录，保留上次证据')
    a.preflight_mode=True
    from hsbg.az.train.personal import write_json,fingerprint,sha256,training_source_digest
    from workflows.run_overnight_learning import run_process
    from workflows.run_learning_experiment import training_report
    from workflows.audit_full_trainer import audit
    from workflows.report_firestone_learning import panel_diagnostics
    deadline=time.monotonic()+35*60
    def run(*args):run_process([sys.executable,*map(str,args)],deadline)
    run('workflows/check_actor_tensor_path.py','--checkpoint',a.source_7357/'model.pt','--device','auto','--require-gpus',2,'--out',a.out/'actor.json')
    parent,source,inherited,factor,paths,manifests,cfg,previous=inputs(a)
    write_json(a.out/'train.config.json',cfg);train=a.out/'train';versions=[]
    for i,key in enumerate(('warm0','warm1','train')):
        extra=['--initial-trainer-state',source/'trainer.pt'] if i==0 else ['--resume','--library-transition-from',manifests[('warm0','warm1','train')[i-1]]['digest']]
        run('-m','hsbg.az.train.personal','train','--config',a.out/'train.config.json','--checkpoint',source/'model.pt','--library',paths[key],'--out',train,'--device','cuda:0','--run-iterations',1,*extra)
        latest=read(train/'latest.json');versions.append(latest['version'])
        if latest['version']!=7358+i:raise ValueError('更新版本不连续')
    quality=training_report(train,a.out/'health',versions[-1]);tot=quality['totals'];strict=quality['mirror_strict_counts']
    if not tot['mirror_attempt']==tot['mirror_applied']==strict['eligible_live_seats'] or any(tot.get(k,0) for k in ('illegal_actions','truncated_games','mirror_errors','mirror_missing','mirror_fallback')):raise ValueError('工程注入验收失败')
    if tot['optimizer_steps']<=0 or tot['processed_samples']/tot['decisions']<.8:raise ValueError('样本有效学习率过低，保留诊断')
    final=audit(train/'snapshots'/f"step_{versions[-1]:06d}",versions[-1],device='cuda:0')
    initial=read(parent/'source-fork.json')['restore_audit']
    if final['adam_step_min']<=initial['adam_step_min']:raise ValueError('Adam未连续推进')
    # 八局只验收诊断链路，不作棋力资格门；正式长训仍有96局冻结起点。
    run('-m','hsbg.az.train.personal','eval','--checkpoint',source/'model.pt','--library',paths['hard'],'--device','cuda:0','--games',1,'--seed',14000000,'--out',a.out/'evaluation.json','--mirror-strict','--economy','--exposure',1,'--max-actions-per-turn',80,'--move-guard',1,'--move-cap',5,'--move-repeat-limit',2)
    evaluation=read(a.out/'evaluation.json');write_json(a.out/'diagnostics.json',panel_diagnostics(evaluation,a.out/'evaluation.json'))
    result=dict(passed=True,runner_sha256=sha256(ROOT/'workflows/run_7357_long.py'),source_sha256=inherited['sha256'],fingerprint=fingerprint(),training_source_sha256=training_source_digest(),libraries={k:v['digest'] for k,v in read(a.bot_3702/'report.json')['splits'].items()},versions=versions,training=quality,final_restore=final,source_adam_step_min=initial['adam_step_min'],probe_weights_not_used_for_long_training=True)
    write_json(a.out/'report.json',result);print(json.dumps({'passed':True,'versions':versions,'training':quality['rates']},ensure_ascii=False))

if __name__=='__main__':main()
