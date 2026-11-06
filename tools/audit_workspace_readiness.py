"""本地完整性与旧来源核对；不删除文件，不接触NUS或游戏窗口。"""
from pathlib import Path
import json,sys,importlib,hashlib
R=Path(__file__).resolve().parents[1]
def main():
 report=dict(workspace=str(R),python=sys.executable,sys_path=sys.path,imports={},old_source_changes=[],missing=[])
 for name in ('hsbg','hsbg_engine','hsrl_agent','torch','numpy','scipy','PIL','cv2','rapidocr','onnxruntime'):
  try:
   m=importlib.import_module(name);report['imports'][name]=dict(path=getattr(m,'__file__',None),version=getattr(m,'__version__',None))
  except ImportError as exc:report['missing'].append(dict(module=name,error=str(exc)))
 report['old_project_path_dependency']=any(any(old.casefold() in str(p).casefold() for old in ('D:/HSBG','D:/HearthstoneAI','D:/HearthstoneRL')) for p in [str(x).replace('\\','/') for x in sys.path]+[str(x['path']).replace('\\','/') for x in report['imports'].values()])
 inv=json.loads((R/'records/migration/2026-11-06/inventory.json').read_text(encoding='utf8'))
 for x in inv['files']:
  p=Path(x['root'])/x['relative']
  try:
   s=p.stat()
   if (s.st_size,s.st_mtime_ns)!=(x['bytes'],x['mtime_ns']):report['old_source_changes'].append(str(p))
  except FileNotFoundError:report['old_source_changes'].append(str(p))
 report['old_files_checked']=len(inv['files'])
 for path in ('engine/Cargo.toml','engine/pyproject.toml','engine/FORK.md','config/engine-upstream-lock.json','config/live/model.json','client/live_245.py','client/start.ps1','deploy/bootstrap_nus.sh'):
  if not (R/path).is_file():report['missing'].append(dict(file=path))
 report['selected_model']=json.loads((R/'config/live/model.json').read_text(encoding='utf8'))
 p=R/report['selected_model']['checkpoint']
 with p.open('rb') as f:report['selected_model_hash_ok']=hashlib.file_digest(f,'sha256').hexdigest()==report['selected_model']['sha256']
 report['local_code_and_model_ready']=not report['missing'] and not report['old_project_path_dependency'] and report['selected_model_hash_ok']
 report['real_game_accepted']=False;report['nus_redeployment_accepted']=False
 report['remaining']=['当前游戏版本界面/时序/真实日志验收','当前37.0.2战团帧和raw源局','24h最终状态和全程日志/终评','原版4159/6157等历史状态','HSAI新布局的NUS分叉迁移及GPU验收','注入前物种账本偏差定位','play/cancel无进展循环处理']
 output=R/'artifacts/local/workspace-completeness.json';output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({k:v for k,v in report.items() if k not in ('imports','sys_path','old_source_changes')},ensure_ascii=False,indent=2))
 return 0 if report['local_code_and_model_ready'] else 2
if __name__=='__main__':raise SystemExit(main())
