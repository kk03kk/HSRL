#!/usr/bin/env bash
set -euo pipefail
root=$1; assets=$2
destination="$root/fresh-bank.jsonl"
tmp="$destination.tmp"
for i in 0 1 2 3; do
    "$assets/warband_collect" 183000000 "$i" 4 3 > "$root/fresh-$i.jsonl"
done
head -1 "$root/fresh-0.jsonl" > "$tmp"
for i in 0 1 2 3; do tail -n +2 "$root/fresh-$i.jsonl" >> "$tmp"; done
mv "$tmp" "$destination"
python3 - "$destination" "$root/fresh-bank-receipt.json" <<'PY'
import json,hashlib,sys
from pathlib import Path
path,out=map(Path,sys.argv[1:])
with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
with path.open() as f:
    header=json.loads(next(f)); masks=set(); n=0
    for line in f:
        row=json.loads(line)
        assert 183000000<=row["seed"]<183100000
        masks.add(row['lobby']); n+=1
assert len(masks)==252
tmp=out.with_suffix(".tmp")
tmp.write_text(json.dumps({"sha256":digest,"tracks":n,"lobbies":len(masks),"header":header}))
tmp.replace(out)
PY
