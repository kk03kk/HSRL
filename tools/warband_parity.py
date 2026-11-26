"""Compare the unmodified script arena against earlier complete engine traces."""
import json
from pathlib import Path
import sys

import numpy as np
import taverngym
from taverngym._core import BcArena

source, target = map(Path,sys.argv[1:])
n_checked = actions = 0
for field in ("original","mixed"):
    with (source/f"trace-{field}.jsonl").open() as f:
        old = [json.loads(line) for line in f][:256]
    arena = BcArena([row["game"]["seed"] for row in old],field=="mixed",8)
    offsets = np.zeros(len(old),dtype=int)
    while True:
        slots,*_ = arena.observe()
        active = np.frombuffer(slots,dtype="<u4")
        if not len(active):
            break
        arena.step([old[i]["actions"][offsets[i]] for i in active])
        offsets[active] += 1
    assert json.loads(arena.results()) == [row["game"] for row in old]
    for i,count in enumerate(offsets):
        assert count == len(old[i]["actions"])
    n_checked += len(old)
    actions += int(sum(offsets))
target.write_text(json.dumps({"games":n_checked,"actions":actions,
            "identical_trajectory":True,"engine":taverngym.build_info()["source_hash"]},indent=2))
