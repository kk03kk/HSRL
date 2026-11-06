"""本地资源整理：只读来源，哈希去重复制/归档，原目录不写入。"""
from pathlib import Path
import os,json,hashlib,zipfile,collections,re,time,shutil
ROOT=Path('D:/HSAI');BASE=ROOT/'records/migration/2026-11-06'
INVENTORY=json.loads((BASE/'inventory.json').read_text(encoding='utf-8'))
CACHE_PARTS={'cargo-target','target','target-gnu','target-llvm','build','dist','site-packages','node_modules','__pycache__','.venv','venv','.pytest_cache','.ruff_cache','live-runtime-deps'}
BINARY={'.pyc','.pyd','.dll','.so','.rlib','.rmeta','.exe','.o','.obj','.lib','.a','.pdb','.whl','.ninja'}
CODE={'.py','.rs','.cs','.csproj','.toml','.lock','.sh','.sbatch','.ps1','.cmd','.bat','.yml','.yaml','.cff'}
MODEL={'.pt','.pth','.ckpt','.safetensors'}
ROOTS={'D:\\HSBG':'hsbg','D:\\HearthstoneAI':'hearthstoneai','D:\\HearthstoneRL':'hearthstonerl'}

def classify(x):
 origin=ROOTS[x['root']];rel=x['relative'];p=Path(rel);parts=p.parts;name=p.name.lower();suffix=p.suffix.lower()
 if any(q in CACHE_PARTS or q.endswith('.egg-info') for q in parts[:-1]) or suffix in BINARY:return 'skip',None,'可重新生成的环境/构建产物'
 auto=origin=='hearthstonerl' and rel.startswith('AutoTavern/')
 sub=rel[len('AutoTavern/'):] if auto else rel
 source_id='autotavern' if auto else origin
 # 可恢复的源代码快照：不进入活动Python路径，不混入另一训练引擎。
 if suffix in CODE or name.startswith(('license','notice','readme','agents','repo_status','project_workspace','project-workspace')) or name in {'.gitignore','.gitattributes','makefile','requirements.txt'}:
  return 'zip','records/source-snapshots/'+source_id+'.zip','原始代码/配置/说明快照'
 if suffix=='.pdf':return 'copy','docs/research/papers/'+p.name,'论文原文'
 # 精确保留所有旧文档和NUS资料，AGENTS改名避免嵌套指令生效。
 if suffix=='.md' or (suffix=='.txt' and ('requirements' in name or 'NUS' in rel)):
  dst='records/source-docs/'+source_id+'/'+sub
  return 'copy',dst,'原始研究/服务器说明'
 if origin=='hsbg' and rel.startswith('personal/artifacts/libraries/'):
  return 'copy','artifacts/libraries/'+rel.split('personal/artifacts/libraries/',1)[1],'完整战团库及指纹'
 if origin=='hsbg' and rel.startswith('personal/local/upstream-artifacts/'):
  return 'copy','artifacts/datasets/friend-reference/'+rel.split('personal/local/upstream-artifacts/',1)[1],'BC/搜索示范数据与来源'
 if origin=='hsbg' and rel.startswith('personal/local/') and parts[2] in {'firestone-20261102','firestone-255790-refresh','firestone','firestone-current','firestone-preview'} and len(parts)>3:
  # 原始真人数据按抓取批次保留；相同文件会通过copy的可复用索引去重。
  return 'raw-firestone','artifacts/datasets/firestone/'+parts[2]+'/'+Path(*parts[3:]).as_posix(),'Firestone原始feed/回放/逐局统计'
 if rel.startswith('FirestoneData/'):
  return 'copy','artifacts/datasets/firestone/source-tools/'+p.name,'用户提供的Firestone入口资源'
 if origin=='hearthstoneai' and rel.startswith(('hsdata/','reversed/')):
  return 'copy','artifacts/datasets/legacy-hsdata/'+rel,'历史卡表与源XML，兼容旧模型'
 if rel.startswith('data/') and not name.startswith('.'):
  return 'copy','artifacts/datasets/legacy-catalogs/'+source_id+'/'+rel[5:],'版本化静态数据或旧战团/题库'
 if auto and sub.startswith('configs/'):
  return 'copy','config/historical/autotavern/'+sub[8:],'旧训练与实机配置'
 if suffix=='.rar' or name.endswith('.tar.gz') or (suffix=='.zip' and not name.endswith('.xml.zip')):
  # 优先保留服务器导出和原始模型压缩包；临时打包工具不迁移。
  if not any(q in rel.lower() for q in ('cargo','site-packages','debug/deps')):
   return 'archive','artifacts/archives/'+source_id+'/'+sub,'原始交付/服务器导出包'
 if suffix in MODEL:
  if origin=='hsbg' and rel.startswith('personal/local/models/'):
   return 'model','artifacts/models/best/'+p.name,'原始s14 Best'
  if origin=='hsbg' and rel.startswith('personal/local/3702-best-migrated/'):
   return 'model','artifacts/models/best/s14-3702/'+p.name,'37.0.2卡表迁移后的Best'
  if origin=='hearthstoneai' and rel=='policy_bc.pt':
   return 'model','artifacts/models/legacy-client/policy_bc.pt','旧候选模型实机BC权重'
  bad=('smoke','fixture','integration','probe','pytest','test_','validation','repair','native','acceptance','migration-test','_weights','_pool_weights')
  important=('/nus-' in '/'+sub or '/model-candidates/' in '/'+sub or '/full-training-audit-20261028/friend/' in '/'+sub or '/full-training-audit-20261028/ours/' in '/'+sub or '/private_research/turn_search_localpair/' in '/'+sub or (origin=='hearthstoneai' and rel.startswith('runs/bc/')))
  if important and not any(q in sub.lower() for q in ('pytest','test_','_weights','_pool_weights')):
   return 'model','artifacts/models/historical/'+source_id+'/'+sub,'正式历史模型/完整恢复状态'
  if any(q in rel.lower() for q in bad) or (origin=='hsbg' and rel.startswith('personal/runs/')):
   return 'skip-model',None,'可重生成的工程冒烟/合成测试权重；保留实验报告'
  return 'model','artifacts/models/historical/'+source_id+'/'+sub,'历史模型待分类保留'
 # 实机原始证据全部保留到只读归档，不激活旧未决输入账本。
 if origin=='hearthstoneai' and rel.startswith('runtime/'):
  return 'zip','records/evidence/live/hearthstoneai/'+parts[1]+'.zip','真实游戏日志/截图/执行回执'
 if auto and sub.startswith('runs/') and any(q in sub.split('/')[1].lower() for q in ('live-','autoplay-','client-inspection')):
  return 'zip','records/evidence/live/autotavern/'+sub.split('/')[1]+'.zip','模型实机原始输入/截图/账本/回执'
 if auto and sub.startswith(('runs/library-stage/','runs/library-life-v3/')) and suffix in {'.json','.gz','.jsonl','.npz','.npy'} and x['bytes']>1024*1024 and 'test_' not in sub:
  return 'copy','artifacts/datasets/legacy-autotavern/'+sub,'旧版战团/示范训练数据'
 # 论文阶段真实实验、性能与故障报告保留；巨型合成状态/帧缓存可重新生成。
 if name=='synthetic.json' or suffix in {'.npy','.npz','.bin','.mmap','.ipc','.dat'}:
  return 'skip',None,'可重生成的合成状态/IPC/采样缓存'
 if suffix in {'.json','.jsonl','.log','.txt','.csv','.xml','.png','.jpg','.jpeg','.gz','.html','.tsv','.sha256'}:
  if x['bytes']>20*1048576 and origin=='hsbg' and rel.startswith('personal/local/'):
   return 'skip',None,'大型本地生成中间产物，保留摘要与重现源码'
  return 'zip','records/evidence/experiments/'+source_id+'.zip','训练/评测/性能/故障原始证据'
 return 'review',None,'未自动判定的资源，保留旧目录等待核实'

if __name__=='__main__':
 rows=[];tot=collections.defaultdict(lambda:[0,0])
 for x in INVENTORY['files']:
  action,destination,reason=classify(x);y={**x,'action':action,'destination':destination,'reason':reason};rows.append(y);tot[action][0]+=1;tot[action][1]+=x['bytes']
 (BASE/'resource-plan.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 print({k:{'files':v[0],'GiB':round(v[1]/1073741824,3)} for k,v in tot.items()})
