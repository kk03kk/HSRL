"""Bind a model to vocabulary order, native action semantics and observation layout.

Compute node / local CPU only. Reports retained checkpoint static-buffer drift;
never rewrites a model's learned or frozen tensors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hsbg.runtime.artifacts import atomic_json,file_identity


def validate(checkpoint):
    import torch
    import hsbg_engine as engine
    from client.hsrl_agent.compat import verify
    torch.set_num_threads(2)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    state=payload['state_dict']
    buffers={key:state[key].cpu().numpy() for key in ('card_static','card_text') if key in state}
    report=verify(buffers)
    if not report['ok']: raise ValueError(report['hard_failures'])
    actual=engine.build_info()['source_hash']
    # The lock schema is independently checked by verify_workspace; retain its
    # content identity rather than guessing a nested source-hash key here.
    actions=[engine.action_name(i) for i in range(245)]
    semantic={'engine_source':actual,'card_ids_sha256':report['cards']['ids_sha256'],
              'layout':report['layout'],'obs_slices':report['obs_slices'],
              'action_names':actions,'static':report['static']}
    return {'schema':'hsai.model-contract-validation/v1','valid':True,
            'checkpoint':file_identity(checkpoint),'engine_lock':file_identity(ROOT/'config/engine-upstream-lock.json'),
            'semantic_sha256':hashlib.sha256(json.dumps(semantic,sort_keys=True).encode()).hexdigest(),
            'semantics':semantic,'limitations':'checkpoint lacks a historical vocabulary digest; current ordered vocabulary is checked against the bundled frozen reference and lineage is recorded separately'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=validate(args.checkpoint)
    atomic_json(args.out,result)
    print(json.dumps({'valid':result['valid'],'static':result['semantics']['static'],'semantic_sha256':result['semantic_sha256']}))
