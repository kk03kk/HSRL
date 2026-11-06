"""本地校验归档、逐文件SHA及7357固定来源；安全解包到独立目录。"""
import argparse, hashlib, json, shutil, tarfile
from pathlib import Path, PurePosixPath

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def unpack(archive, checksum, output, spec):
    archive=Path(archive); output=Path(output)
    expected=Path(checksum).read_text(encoding='ascii').split()[0]
    if len(expected)!=64 or sha(archive)!=expected: raise ValueError('Archive SHA256 mismatch; retained for inspection')
    required={i['destination']:i for i in spec['items']}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers(); names=[m.name for m in members]
        if len(names)!=len(set(names)) or any(not m.isfile() for m in members): raise ValueError('Duplicate/non-file archive member')
        meta=tar.getmember('bundle-manifest.json')
        if meta.size>1024*1024: raise ValueError('Manifest too large')
        manifest=json.load(tar.extractfile(meta)); records=manifest['files']
        paths=[r['path'] for r in records]
        if len(paths)!=len(set(paths)) or set(paths)!=set(required): raise ValueError('Unexpected or missing manifest files')
        if set(names)!={'bundle-manifest.json'}|{'payload/'+p for p in paths}: raise ValueError('Archive member set mismatch')
        output.mkdir(parents=True,exist_ok=True)
        for row in records:
            rel=PurePosixPath(row['path'])
            if rel.is_absolute() or '..' in rel.parts or '\\' in str(rel) or ':' in str(rel): raise ValueError('Unsafe path')
            item=required[row['path']]
            if item.get('sha256') and row['sha256']!=item['sha256']: raise ValueError('Pinned7357 hash differs')
            member=tar.getmember('payload/'+row['path'])
            if member.size!=row['bytes']: raise ValueError('Member size mismatch')
            dest=output.joinpath(*rel.parts)
            if dest.exists():
                if sha(dest)==row['sha256']: continue
                raise ValueError('Existing file differs; retained: '+str(dest))
            dest.parent.mkdir(parents=True,exist_ok=True)
            partial=dest.with_name(dest.name+'.unpack-partial')
            with tar.extractfile(member) as src,partial.open('wb') as dst: shutil.copyfileobj(src,dst,1024*1024)
            if sha(partial)!=row['sha256']: raise ValueError('File SHA mismatch: '+str(dest))
            partial.rename(dest)
        receipt=dict(schema='hsai.closeout_capture/v2',job=1080772,archive_sha256=expected,files=records,download_complete=True,trainer_semantics_verified=False)
        (output/'capture.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Archive and all files verified; extracted:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True,type=Path);p.add_argument('--checksum',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    spec=json.loads((Path(__file__).resolve().parents[1]/'config/nus-1080772-closeout-download.json').read_text(encoding='utf-8'))
    unpack(a.archive,a.checksum,a.out,spec)
