import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--models', required=True)
    p.add_argument('--library', required=True)
    p.add_argument('--games', type=int, default=12)
    p.add_argument('--seed', type=int, default=21000000)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path,
                   default=ROOT / 'artifacts/models/leaderboard-v1/manifest.json')
    a = p.parse_args()

    manifest = json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    stable = {m['id']: m for m in manifest['stable']}
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    library = Path(a.library).resolve()

    index = dict(schema='hsai.leaderboard_eval/v1', library=str(library),
                 games=int(a.games), seed=int(a.seed), device=a.device,
                 models=[])
    for mid in [x.strip() for x in a.models.split(',') if x.strip()]:
        entry = stable.get(mid)
        row = dict(id=mid)
        if entry is None:
            row['error'] = 'not in stable manifest'
            index['models'].append(row)
            (out / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
            continue
        checkpoint = (ROOT / entry['model_path']).resolve()
        result_path = out / f'{mid}.json'
        row.update(version=entry.get('version'), cum_decisions=entry.get('cum_decisions'),
                   sha256=entry.get('sha256'), checkpoint=str(checkpoint),
                   result=str(result_path))
        cmd = [sys.executable, '-m', 'hsbg.az.train.personal', 'eval',
               '--checkpoint', str(checkpoint), '--library', str(library),
               '--device', a.device, '--games', str(a.games), '--seed', str(a.seed),
               '--out', str(result_path), '--mirror-strict', '--exposure', '1',
               '--economy', '--max-actions-per-turn', '80', '--move-guard', '1',
               '--move-cap', '5', '--move-repeat-limit', '2', '--policy', 'net']
        print(f'[leaderboard] {mid} version={entry.get("version")} games={a.games}', flush=True)
        started = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        row['seconds'] = round(time.time() - started, 2)
        row['returncode'] = proc.returncode
        if proc.returncode:
            row['stderr_tail'] = proc.stderr[-3000:]
            row['stdout_tail'] = proc.stdout[-1000:]
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding='utf-8'))
                row.update(valid=result.get('valid'), mean_rank=result.get('mean_rank'),
                           mean_rank_ci95=result.get('mean_rank_ci95'),
                           top4_rate=result.get('top4_rate'), win_rate=result.get('win_rate'),
                           protocol_sha256=result.get('protocol_sha256'))
            except Exception as exc:
                row['result_error'] = repr(exc)
        else:
            row['result_error'] = 'result file missing'
        index['models'].append(row)
        (out / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[leaderboard] {mid} rc={proc.returncode} valid={row.get("valid")} '
              f'rank={row.get("mean_rank")} top4={row.get("top4_rate")} win={row.get("win_rate")}', flush=True)
    print(json.dumps(index, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
