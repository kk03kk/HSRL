"""Capture allocated Slurm resources for a bounded, documented research window.

Runs locally; the remote command is accounting only. GPU units are allocated
slices, not equivalent physical H100/H200 GPU-hours or measured utilization.
"""
import argparse,datetime,json,re,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.hsai_ops import remote_run
from hsbg.runtime.artifacts import atomic_json,file_identity

FIELDS=('JobIDRaw','JobName','State','ExitCode','ElapsedRaw','AllocCPUS','AllocTRES','Start','End','Submit','Partition')


def totals(rows):
    gpu_seconds=0;cpu_seconds=0;states=Counter()
    for row in rows:
        elapsed=int(row['ElapsedRaw'] or 0)
        match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',row['AllocTRES'])
        gpu_seconds+=elapsed*(int(match.group(1)) if match else 0)
        cpu_seconds+=elapsed*int(row['AllocCPUS'] or 0)
        states[row['State'].split()[0]]+=1
    return {'jobs':len(rows),'allocated_gpu_slice_hours':gpu_seconds/3600,
        'allocated_cpu_core_hours':cpu_seconds/3600,'states':dict(states)}


def main(args):
    datetime.datetime.fromisoformat(args.since)
    if not re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d',args.since):
        raise ValueError('since must be an explicit local Slurm date/time')
    # No job mutation, process inspection, GPU import, or trace scan on login.
    rc,out,err=remote_run('sacct --allocations -n -P -S '+args.since+
        ' --format='+','.join(FIELDS),check=True,label='research-accounting')
    def parse(output):
        rows=[]
        for line in output.splitlines():
            values=line.split('|')
            if not values or not values[0]:continue
            if len(values)==len(FIELDS)+1 and not values[-1]:values.pop()
            if len(values)!=len(FIELDS):raise ValueError('accounting field mismatch')
            row=dict(zip(FIELDS,values))
            if not row['JobIDRaw'].isdigit():raise ValueError('unexpected accounting step row')
            rows.append(row)
        return rows
    raw=parse(out)
    docs=[Path('docs/analysis/AUTONOMOUS_RESEARCH_20261118.md'),Path('docs/DECISIONS.md')]
    documented={int(x) for p in docs for x in re.findall(r'(?<![0-9])11[0-9]{5}(?![0-9])',p.read_text(encoding='utf-8'))}
    # Site sacct -S omits some never-started jobs. Query explicit documented
    # and live queue IDs too, retaining cancellations and blocked dependencies.
    _,queued,_=remote_run('squeue -h -u "$USER" -o "%A"',check=True,label='research-accounting-queue')
    lower=min((int(r['JobIDRaw']) for r in raw if r['JobName'].startswith('hsai-') and r['Submit']>=args.since),default=0)
    extra={i for i in documented if i>=lower}|{int(s) for s in queued.split() if s.isdigit()}
    extra-={int(r['JobIDRaw']) for r in raw}
    if extra:
        _,added,_=remote_run('sacct --allocations -n -P -j '+','.join(map(str,sorted(extra)))+
            ' --format='+','.join(FIELDS),check=True,label='research-accounting-never-started')
        raw+=parse(added)
    rows=sorted((r for r in raw if r['JobName'].startswith('hsai-') and r['Submit']>=args.since),key=lambda r:int(r['JobIDRaw']))
    # The requested window, account ownership enforced by sacct, name prefix,
    # and submit time define inclusion; documentation gaps remain visible.
    result={'queried_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'slurm_local_since':args.since,'slurm_timezone':'Asia/Singapore',
        'scope':'Current account hsai-* allocations submitted during this research window; pending and failed retained',
        'rows':rows,'totals':totals(rows),'documentation':{str(p):file_identity(p) for p in docs},
        'jobs_not_yet_in_report_or_decisions':[r['JobIDRaw'] for r in rows if int(r['JobIDRaw']) not in documented],
        'limitations':['Running allocations are incomplete snapshots',
            'GPU count is allocated MIG slices, not physical GPU equivalence or utilization',
            'CPU core-hours include reserved idle time; not wall-clock research time']}
    path=Path(args.out)
    if path.exists():raise FileExistsError(path)
    atomic_json(path,result);print(json.dumps(result['totals']),flush=True)
    print(json.dumps({'undocumented':result['jobs_not_yet_in_report_or_decisions']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--since',required=True)
    p.add_argument('--out',required=True);main(p.parse_args())
