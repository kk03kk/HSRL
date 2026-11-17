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
CODE_DIRS=('hsbg','workflows','tests','tools','client','deploy','ops')


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


# ── 仓库卫生门禁（2026-11-17 审计 P1-7：规则写了但既没门禁也没人执行）──────────
#: 根目录**只允许**这些文件（`AGENTS.md` §2）。目录不受限（`.venv/`、`artifacts/` 等
#: 由 `.gitignore` 管）。
ROOT_ALLOWED_FILES=frozenset({
    'README.md','AGENTS.md','pyproject.toml','requirements.txt',
    '.gitignore','.gitattributes','.git',  # worktree .git is a pointer file
})
#: 单个文件进 Git 的上限（`AGENTS.md` §2）
BIG_FILE_LIMIT=5*1024*1024


def check_root_files(root=ROOT):
    """根目录白名单：返回违规的**文件名**列表（只看文件，不看目录）。"""
    bad=[]
    for p in sorted(Path(root).iterdir()):
        if p.is_file() and p.name not in ROOT_ALLOWED_FILES:
            bad.append(p.name)
    return bad


def git_tracked_files(root=ROOT):
    """``git ls-files`` 的结果；git 不可用时返回 ``None``（跳过该门禁而不是误报）。"""
    try:
        out=subprocess.run(['git','ls-files','-z'],cwd=str(root),capture_output=True,
                           text=True,encoding='utf-8',errors='replace',timeout=120)
    except (OSError,subprocess.SubprocessError):
        return None
    if out.returncode!=0:return None
    return [n for n in out.stdout.split('\0') if n]


def check_big_tracked(root=ROOT,limit=BIG_FILE_LIMIT):
    """**已进 Git** 的单文件大小上限（权重/日志/压缩包不应入库）。"""
    names=git_tracked_files(root)
    if names is None:return []
    bad=[]
    for rel in names:
        p=Path(root)/rel
        try:
            size=p.stat().st_size
        except OSError:
            continue
        if size>limit:
            bad.append(f'{rel} ({size/1024/1024:.1f} MB > {limit/1024/1024:.0f} MB)')
    return bad


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
 # ── 仓库卫生（2026-11-17 审计 P1-7）：根目录白名单 + 进 Git 的单文件大小上限。
 # 以前这两条规则只写在 AGENTS.md §2 里，没有任何门禁真正执行。
 root_bad=check_root_files(ROOT)
 big_bad=check_big_tracked(ROOT)
 gates.append(dict(name='root-whitelist',passed=not root_bad,seconds=0.0,
                   stdout='' ,stderr='' if not root_bad else
                   '根目录只允许 README.md/AGENTS.md/pyproject.toml/requirements.txt/'
                   '.gitignore/.gitattributes；实际多出: '+', '.join(root_bad)))
 gates.append(dict(name='big-file',passed=not big_bad,seconds=0.0,
                   stdout='',stderr='' if not big_bad else
                   '单文件进 Git 上限 5 MB；超限: '+', '.join(big_bad)))
 for g in gates:
  print('[PASS]' if g['passed'] else '[FAIL]',g['name'],flush=True)
  if not g['passed'] and g['stderr']:print('  ',g['stderr'],flush=True)
 commands=[('cards',[sys.executable,'tools/p2_data/check_carddb_3702.py'],120),
           ('engine-source',[sys.executable,'tools/p2_data/check_engine_upstream.py'],120)]
 if not a.skip_rust:
  commands.append(('rust',['cargo','test','--release','--manifest-path','engine/Cargo.toml','-p','hsbg-engine'],a.rust_timeout))
 commands.append(('python',[sys.executable,'-m','pytest','tests','-q','-p','no:cacheprovider'],a.python_timeout))
 for name,command,timeout in commands:
  print('[RUN]',name,flush=True);start=time.monotonic()
  result=subprocess.run(command,cwd=ROOT,timeout=timeout,text=True,encoding='utf-8',errors='replace',capture_output=True)
  # `pytest` 在"一个测试都没收集到"时返回 5：那也是失败（以前容易被当成通过）。
  ok=result.returncode==0
  gates.append(dict(name=name,passed=ok,seconds=round(time.monotonic()-start,2),stdout=result.stdout,stderr=result.stderr))
  print('[PASS]' if ok else '[FAIL]',name,flush=True)
  if not ok and name=='python' and result.returncode==5:
   print('  pytest 没有收集到任何测试（rc=5）——检查 tests/ 是否被整模块跳过/写坏',flush=True)
 a.json.parent.mkdir(parents=True,exist_ok=True)
 report=dict(passed=all(g['passed'] for g in gates),gates=gates)
 a.json.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
