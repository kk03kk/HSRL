"""查找旧资源对应的新位置；只读，不自动解压或重放任何历史输入。"""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('query',help='源路径、文件名、SHA或作业号');a=p.parse_args()
 for name in ('migration-map.json','supplemental-map.json','model-hashes.json'):
  f=ROOT/'records/migration/2026-11-06'/name
  if not f.exists():continue
  for r in json.loads(f.read_text(encoding='utf-8')):
   if a.query.casefold() in json.dumps(r,ensure_ascii=False).casefold():
    if r.get('destination'):r['absolute_destination']=str(ROOT/r['destination'])
    print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
