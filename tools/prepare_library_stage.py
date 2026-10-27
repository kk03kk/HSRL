"""Generate source-disjoint native libraries and bind runtime configs to checksums."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
try:
    from tools.local_offline import prepare
except ModuleNotFoundError:
    from local_offline import prepare


def prepare_stage(config, out, counts, manual=None, *, required_contexts=None):
    import taverngym as tg
    from taverngym.research.library_build import generate
    from taverngym.rl.library_bank import assert_disjoint
    from taverngym.research.library_contexts import expected_contexts, seed_plan
    from taverngym.rl.config import Config
    cfg = Config.load(config)
    required = expected_contexts(cfg.rules) if required_contexts is None else set(required_contexts)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('prepare requires a new dataset directory')
    out.mkdir(parents=True)
    reports = {}
    for split, games, seed in zip(('train','validation','test'), counts, (2810000,3810000,4810000)):
        copies = max(1, math.ceil(games/len(required)))
        plan = seed_plan(cfg.rules, required, start=seed, copies=copies)
        print(json.dumps({'phase':'context_plan','split':split,'contexts':len(required),'source_games':len(plan)}),flush=True)
        reports[split] = generate(config,out/f'{split}.json',seed=seed,split=split,game_seeds=plan)
        bank_path = out/f'{split}.json'
        bank = json.loads(bank_path.read_text(encoding='utf-8'))
        for attempt in range(3):
            covered = {(t['lobby'],t['deity'],t['paradox']) for t in bank['traces']}
            missing = required-covered
            if not missing:
                break
            # A protected/truncated source is discarded, never ranked as a normal
            # game. Replace only its missing context from a disjoint seed range.
            extra_plan = seed_plan(cfg.rules, missing, start=seed+(attempt+1)*250000)
            extra_path = out/f'{split}-replacement-{attempt}.json'
            extra = generate(config,extra_path,seed=seed+(attempt+1)*250000,split=split,
                             game_seeds=extra_plan,allow_empty=True)
            addition = json.loads(extra_path.read_text(encoding='utf-8'))
            bank['traces'].extend(addition['traces'])
            reports[split]['rejected'].extend(extra['rejected'])
            reports[split]['source_games'] += extra['source_games']
            reports[split]['requested_games'] += extra['requested_games']
            reports[split]['seconds'] += extra['seconds']
        covered = {(t['lobby'],t['deity'],t['paradox']) for t in bank['traces']}
        if covered != required:
            raise RuntimeError(f'{split}: {len(required-covered)} native contexts still lack valid sources')
        raw = json.dumps(bank,separators=(',',':')).encode()
        bank_path.write_bytes(raw)
        reports[split].update(sha256=hashlib.sha256(raw).hexdigest(),required_contexts=len(required),
                             covered_contexts=len(covered),seat_traces=len(bank['traces']),
                             frames=sum(len(t['frames']) for t in bank['traces']))
        sources = {t['source_group']:(t['lobby'],t['deity'],t['paradox']) for t in bank['traces']}
        reports[split]['contexts'] = dict(Counter(str(k) for k in sources.values()))
        (out/f'{split}.report.json').write_text(json.dumps(reports[split],indent=2),encoding='utf-8')
        del bank
    if manual:
        trace = json.loads(Path(manual).read_text(encoding='utf-8'))
        bank_path = out/'train.json'
        bank = json.loads(bank_path.read_text())
        if trace['card_table'] != bank['card_table']:
            raise ValueError('manual trace card table differs')
        compatible = [t for t in bank['traces'] if all(t[k] == trace[k] for k in ('lobby','deity','paradox'))]
        if not compatible:
            raise ValueError('manual proxy context absent from generated training bank')
        trace = {k: v for k, v in trace.items() if k != 'card_table'}
        trace['weight'] = max(1, len(compatible)//9)
        bank['traces'].append(trace)
        bank['manual_notice'] = 'One approximate observed trajectory; weighted near 10% only in its matching context; no independent human test claim.'
        bank_path.write_text(json.dumps(bank,separators=(',',':')),encoding='utf-8')
        reports['train']['sha256'] = hashlib.sha256(bank_path.read_bytes()).hexdigest()
        reports['train']['manual_source_groups'] = 1
        reports['train']['bot_seat_traces'] = reports['train']['seat_traces']
        reports['train']['seat_traces'] = len(bank['traces'])
        reports['train']['frames'] += len(trace['frames'])
        reports['train']['manual_weight_in_matching_context'] = trace['weight']/(trace['weight']+len(compatible))
        (out/'train.report.json').write_text(json.dumps(reports['train'],indent=2),encoding='utf-8')
    assert_disjoint([out/f'{s}.json' for s in reports])
    # Strict matching now covers the entire original random lobby distribution.
    from taverngym.rl.config import Config
    from taverngym.rl.library_bank import bank_text
    cfg = Config.load(config)
    for split, report in reports.items():
        text = bank_text(str(out/f'{split}.json'),report['sha256'])
        check = tg.BatchEnv(16,seed=90711,controlled=range(8),config=cfg.rules,threads=cfg.threads)
        check.auto_reset = False
        if required_contexts is not None:
            # Small local tests may exercise a deliberately limited panel. The
            # production path above always covers every native random context.
            for i in range(16):
                slot = json.loads(check.save(i))
                slot['game'] = json.loads(tg.Game(plan[i % len(plan)],json.dumps(cfg.rules)).state_json())
                check.load(i,json.dumps(slot))
        check.bind_library(text,[g % 8 for g in range(16)])
    runtime = cfg.to_dict()
    runtime.update(opponent_mode='library',library_bank_path=str(out/'train.json'),
                   library_bank_sha256=reports['train']['sha256'],
                   library_validation_path=str(out/'validation.json'),
                   library_validation_sha256=reports['validation']['sha256'])
    for label, split in (('train','validation'),('test','test')):
        c = dict(runtime)
        c.update(library_validation_path=str(out/f'{split}.json'),library_validation_sha256=reports[split]['sha256'])
        (out/f'{label}-config.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    report = {'format':'taverngym.library-preparation.v1','native':tg.build_info(),
              'banks':reports,'source_splits_disjoint':True,'contexts_matched':True,
              'required_contexts':len(required),'full_native_context_coverage':required_contexts is None,
              'manual_proxy':str(manual) if manual else None,'human_calibrated':False,
              'pool_policy':'external-copies-exempt','tail_policy':'hold-final-observed-frame'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    handles = prepare()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--train-games',type=int,default=256)
    p.add_argument('--validation-games',type=int,default=96)
    p.add_argument('--test-games',type=int,default=96)
    p.add_argument('--manual')
    a = p.parse_args()
    print(json.dumps(prepare_stage(a.config,a.out,(a.train_games,a.validation_games,a.test_games),a.manual)))
