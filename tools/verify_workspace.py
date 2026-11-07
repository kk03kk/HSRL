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

#: UTF-8 BOM。代码目录里**一律拒绝**：它会改变源码字节（进而改变
#: ``training_source_sha256`` / ``runner_sha256``）并让 "utf-8 读 + ast.parse"
#: 报假 SyntaxError（2026-11-07 作业 1092598）。
BOM=b'\xef\xbb\xbf'
CODE_DIRS=('hsbg','workflows','tests','tools','client','deploy')


def iter_sources(root=ROOT):
    """受管源码集合：语法检查与 BOM 检查用同一份清单，避免"查了这个漏了那个"。"""
    paths=[]
    for name in CODE_DIRS:
        base=Path(root)/name
        if base.exists():paths.extend(base.rglob('*.py'))
    return sorted(set(paths))


def read_source(path):
    """按 ``utf-8-sig`` 读：BOM 是否允许由 BOM 门禁决定，不该变成假语法错误。"""
    return Path(path).read_text(encoding='utf-8-sig')


def check_sources(root=ROOT):
    """返回 ``(bom_rows, syntax_rows)``：两个列表都为空才算干净。"""
    bom_rows=[];syntax_rows=[]
    for path in iter_sources(root):
        rel=Path(path).relative_to(Path(root)).as_posix()
        try:
            with open(path,'rb') as stream:
                if stream.read(3)==BOM:bom_rows.append(rel)
        except OSError as exc:
            syntax_rows.append(f'{rel}: 读不了 ({exc})');continue
        try:
            ast.parse(read_source(path),filename=rel)
        except SyntaxError as exc:
            syntax_rows.append(f'{rel}:{exc.lineno}: {exc.msg}')
    return bom_rows,syntax_rows


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--rust-timeout',type=int,default=3600)
 p.add_argument('--python-timeout',type=int,default=1800)
 p.add_argument('--json',type=Path,default=ROOT/'artifacts/local/validation.json')
 p.add_argument('--skip-rust',action='store_true')
 a=p.parse_args()
 # 源码门禁先跑：语法 + 代码目录无 BOM。失败就别再花时间构建和跑测试。
 bom_rows,syntax_rows=check_sources(ROOT)
 problems=[f'syntax {row}' for row in syntax_rows]+[f'bom {row}' for row in bom_rows]
 hint='用 `python tools/strip_source_bom.py --fix` 去掉 BOM；不要手改编码，也不要让检查器静默跳过'
 gates=[dict(name='sources',passed=not problems,seconds=0.0,stdout='',
             stderr='' if not problems else '\n'.join(problems)+'\n'+hint)]
 if problems:
  for row in problems:print('[FAIL] source',row,flush=True)
  print('[FAIL] sources:',hint,flush=True)
  a.json.parent.mkdir(parents=True,exist_ok=True)
  a.json.write_text(json.dumps(dict(passed=False,gates=gates),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  return 1
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
