"""引擎保护性结束不是正常胜负：采集/评测/建库共用失败关闭门。"""
import gzip
import json
import os
from pathlib import Path
import time

KEYS=('truncated_combats','truncated_recruit_contexts','turn_limit_end','engine_stopped')


def diagnostics(game):
    if hasattr(game,'diagnostic_counts'):
        values=game.diagnostic_counts()
    else:
        # 兼容历史绑定及测试替身；禁止把读取失败当成零。
        state=json.loads(game.state_json())
        values=(state['truncated_combats'],state.get('truncated_recruit_contexts',0),
                state['turn_limit_end'],bool(state.get('stopped_reason')))
    return dict(zip(KEYS,map(int,values),strict=True))


def require_clean_engine(game):
    counts=diagnostics(game)
    if any(counts.values()):
        out=Path(os.environ.get('HSRL_FAILURE_DIR','artifacts/local/engine-failures'))
        out.mkdir(parents=True,exist_ok=True)
        path=out/f'engine-quality-{os.getpid()}-{time.time_ns()}.json.gz'
        with gzip.open(path,'wt',encoding='utf-8') as stream:
            json.dump(dict(diagnostics=counts,state=json.loads(game.state_json())),stream)
        raise RuntimeError(f'引擎非正常结束，拒绝作为有效训练/评测数据：{counts}；现场：{path}')
    return counts
