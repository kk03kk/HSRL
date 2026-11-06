"""验证新工作区的源码/数据/接口；重任务必须在本地或Slurm计算节点执行。"""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--rust-timeout',type=int,default=3600)
 p.add_argument('--python-timeout',type=int,default=1800)
 p.add_argument('--json',type=Path,default=ROOT/'artifacts/local/validation.json')
 p.add_argument('--skip-rust',action='store_true')
 a=p.parse_args()
 gates=[]
 for path in list((ROOT/'hsbg').rglob('*.py'))+list((ROOT/'workflows').glob('*.py'))+list((ROOT/'client').rglob('*.py')):
  ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
 commands=[('cards',[sys.executable,'tools/p2_data/check_carddb_3702.py'],120),
           ('engine-source',[sys.executable,'tools/p2_data/check_engine_upstream.py'],120)]
 if not a.skip_rust:commands.append(('rust',['cargo','test','--release','--manifest-path','engine/Cargo.toml','-p','hsbg-engine'],a.rust_timeout))
 commands.append(('python',[sys.executable,'-m','pytest','tests','-q','-p','no:cacheprovider'],a.python_timeout))
 for name,command,timeout in commands:
  print('[RUN]',name,flush=True);start=time.monotonic()
  result=subprocess.run(command,cwd=ROOT,timeout=timeout,text=True,encoding='utf-8',errors='replace',capture_output=True)
  gates.append(dict(name=name,passed=result.returncode==0,seconds=round(time.monotonic()-start,2),stdout=result.stdout,stderr=result.stderr))
  print('[PASS]' if result.returncode==0 else '[FAIL]',name,flush=True)
 a.json.parent.mkdir(parents=True,exist_ok=True)
 report=dict(passed=all(g['passed'] for g in gates),gates=gates)
 a.json.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
