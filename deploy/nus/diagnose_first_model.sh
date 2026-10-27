#!/usr/bin/env bash
# Upload this script and diagnose_pilot.py to HOME. Submit one CPU read-only audit.
set -euo pipefail
project="${HSRL_PROJECT:-$HOME/projects/hearthstone-rl/AutoTavern/runs/releases/paper-stage-v1/AutoTavern}"
run="$project/runs/nus-paper-pilot-20261027-132048/compact_gru128-s701"
cd "$project"
python3 tools/package_nus.py --verify .
test -f "$run/latest.pt"
test -f "$run/failure-seed-14001121-game-14.json"
test -f "$HOME/diagnose_pilot.py"
audit="$project/runs/first-model-audit-$(date +%Y%m%d-%H%M%S)"
if [[ -e "$audit" ]]; then printf 'Audit directory exists; stop.\n'; exit 1; fi
mkdir -p "$audit"
cp "$HOME/diagnose_pilot.py" "$audit/diagnose_pilot.py"
cat > "$audit/audit.sbatch" <<'AUDIT_JOB'
#!/usr/bin/env bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
python tools/package_nus.py --verify .
# No build, pip install, optimizer update, or replacement of the accepted release.
set +e
python "$HSRL_AUDIT/diagnose_pilot.py" --run "$HSRL_RUN" --out "$HSRL_AUDIT/results" --games 2
status=$?
set -e
# Keep the diagnostic evidence even if a test/model panel itself failed.
python3 - "$HSRL_RUN" "$HSRL_AUDIT" <<'BUNDLE'
import hashlib,json,sys,tarfile,shutil
from pathlib import Path
run,audit=map(Path,sys.argv[1:])
files={}
for name in ('bc-anchor.pt','latest.pt','best.pt','manifest.json','metrics.jsonl'):
    p=run/name
    if p.exists():files['training/'+name]=p
for p in run.glob('failure-*.json'):files['training/'+p.name]=p
for p in audit.rglob('*'):
    if p.is_file() and p.suffix not in ('.gz','.sha256'):
        files['audit/'+str(p.relative_to(audit))]=p
archive=audit/'first-model-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for name,p in sorted(files.items()):t.add(p,arcname=name,recursive=False)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n')
export=Path.home()/'AutoTavern-first-model-evidence-1023992.tar.gz'
try:
    with export.open('xb') as dest,archive.open('rb') as src:shutil.copyfileobj(src,dest)
    export.with_name(export.name+'.sha256').write_text(digest+'  '+export.name+'\n')
    print('Evidence export: '+str(export),flush=True)
except FileExistsError:
    print('Existing HOME evidence preserved; use this job archive: '+str(archive),flush=True)
print(json.dumps({'archive':str(archive),'sha256':digest,'files':len(files)}),flush=True)
BUNDLE
exit "$status"
AUDIT_JOB
job=$(sbatch --parsable --partition=normal --account=allusers --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G --time=00:45:00 \
  --job-name=hsrl-first-audit --output="$audit/slurm-%j.out" \
  --export="ALL,HSRL_RUN=$run,HSRL_AUDIT=$audit" "$audit/audit.sbatch")
printf 'audit_job=%s\naudit_directory=%s\n' "$job" "$audit"
printf '%s\n' "$job" > "$audit/job.txt"
squeue -j "${job%%;*}"
