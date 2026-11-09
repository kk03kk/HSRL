#!/usr/bin/env python3
"""Bounded A/B runner for the 1095539 follow-up research (no formal long-run submission)."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / 'config/ab-base.json'


def run(argv):
    print('[ab]', ' '.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-dir', required=True)
    p.add_argument('--source-name', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--arm', required=True, choices=['base', 'entropy', 'shaping', 'realbot'])
    p.add_argument('--library', required=True)
    p.add_argument('--eval-library', required=True)
    p.add_argument('--minutes', type=float, default=30.0)
    p.add_argument('--new-decisions', type=int, default=15_000_000)
    p.add_argument('--games', type=int, default=512)
    p.add_argument('--eval-games', type=int, default=64)
    p.add_argument('--seed', type=int, default=200000000)
    p.add_argument('--eval-seed', type=int, default=21000000)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()

    source = Path(a.source_dir).resolve()
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'output exists and is not empty: {out}')
    model = source / 'model.pt'
    trainer = source / 'trainer.pt'
    if not model.is_file() or not trainer.is_file():
        raise SystemExit(f'missing source model/trainer under {source}')
    cfg = json.loads(BASE_CONFIG.read_text())
    cfg['minutes'] = float(a.minutes)
    cfg['games'] = int(a.games)
    cfg['seed'] = int(a.seed)
    if a.arm == 'entropy':
        cfg['entropy-coef'] = 0.001
    elif a.arm == 'shaping':
        cfg['entropy-coef'] = 0.001
        cfg['shaping-coef'] = 0.03
        cfg['shaping-mode'] = 'log_strength'
        cfg['shaping-w-str'] = 1.0
        cfg['shaping-w-tier'] = 0.0
    elif a.arm == 'realbot':
        cfg['entropy-coef'] = 0.001
    # arm base keeps 0.008, no shaping
    base = json.loads((source / 'checkpoint.json').read_text())['actual_decisions']
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / 'train.config.json'
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    cmd = [sys.executable, 'workflows/train_entity_sync.py',
           '--checkpoint', model, '--trainer-state', trainer, '--out', out / 'train',
           '--device', a.device, '--opponent', 'bot', '--behavior', 'net',
           '--pipeline', '1', '--rust-collector', '0', '--lockstep', '1',
           '--eval-panel', 'bot', '--eval-every-decisions', '0',
           '--mirror-snaps', Path(a.library).resolve(), '--mirror-seats', '7',
           '--mirror-p', '1', '--mirror-strict', '1', '--human-board-p', '0',
           '--stop-at-decisions', str(int(base) + int(a.new_decisions)),
           '--minutes', str(a.minutes)]
    for k, v in cfg.items():
        if k in {'games', 'minutes', 'seed'}:
            cmd += ['--' + k, str(v)]
        else:
            cmd += ['--' + k, str(v)]
    run(cmd)
    result = out / 'eval.json'
    run([sys.executable, '-m', 'hsbg.az.train.personal', 'eval',
         '--checkpoint', out / 'train/model.pt', '--library', Path(a.eval_library).resolve(),
         '--device', a.device, '--games', str(a.eval_games), '--seed', str(a.eval_seed),
         '--out', result, '--mirror-strict', '--exposure', '1', '--economy',
         '--max-actions-per-turn', '80', '--move-guard', '1', '--move-cap', '5',
         '--move-repeat-limit', '2', '--policy', 'net'])
    r = json.loads(result.read_text())
    summary = dict(arm=a.arm, source=str(source), source_name=a.source_name,
                   library=str(Path(a.library).resolve()), eval_library=str(Path(a.eval_library).resolve()),
                   config=str(cfg_path), model=str(out / 'train/model.pt'),
                   mean_rank=r.get('mean_rank'), top4_rate=r.get('top4_rate'),
                   win_rate=r.get('win_rate'), protocol_sha256=r.get('protocol_sha256'),
                   elapsed_s=time.time())
    (out / 'ab-result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
