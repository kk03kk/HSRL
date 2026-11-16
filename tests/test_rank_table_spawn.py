"""静态核验：整张名次效用表（只奖励前二）能传到 spawn 采集子进程。"""
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = [1.00, 0.35, -1.00, -1.15, -1.30, -1.45, -1.60, -1.75]


def child(q):
    sys.path.insert(0, ROOT)
    from hsbg.az.train.common import rank_utility_table
    q.put([round(float(x), 3) for x in rank_utility_table()])


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    from hsbg.az.train.common import rank_utility_table, set_rank_utility_table, set_win_bonus

    set_win_bonus(0.7)                       # 先设奖金
    set_rank_utility_table(TABLE)            # 整表覆盖
    os.environ["HSAI_RANK_UTILITY"] = ",".join("%g" % v for v in TABLE)
    parent = [round(float(x), 3) for x in rank_utility_table()]
    print("父进程:", parent)
    assert parent == [round(v, 3) for v in TABLE], "整表没覆盖 win_bonus"

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=child, args=(q,))
    p.start()
    got = q.get(timeout=120)
    p.join(timeout=120)
    print("子进程:", got)
    ok = got == [round(v, 3) for v in TABLE]
    print("TABLE_SPAWN_OK" if ok else "TABLE_SPAWN_FAIL")
    raise SystemExit(0 if ok else 1)
