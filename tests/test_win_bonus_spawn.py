"""验证冠军奖金能传进 spawn 子进程（这是 v13 奖励改动的关键正确性前提）。

不需要 GPU：只起一个 spawn 子进程，让它 import common 并打印名次效用表。
"""
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def child(q):
    sys.path.insert(0, ROOT)
    from hsbg.az.train.common import WIN_BONUS, rank_utility_table
    q.put((WORLD_BONUS if False else None, WIN_BONUS,
           [round(float(x), 3) for x in rank_utility_table()]))


WORLD_BONUS = None

if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    from hsbg.az.train.common import WIN_BONUS as PARENT_B, rank_utility_table, set_win_bonus

    print("父进程初始 :", [round(float(x), 3) for x in rank_utility_table()])
    set_win_bonus(0.7)
    os.environ["HSAI_WIN_BONUS"] = "0.7"
    print("父进程 b=0.7:", [round(float(x), 3) for x in rank_utility_table()])

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=child, args=(q,))
    p.start()
    _, child_b, child_tbl = q.get(timeout=120)
    p.join(timeout=120)

    print("子进程(spawn): b=%s 表=%s" % (child_b, child_tbl))
    ok = abs(child_b - 0.7) < 1e-9 and abs(child_tbl[0] - 1.7) < 1e-6
    print("SPAWN_PROPAGATION_OK" if ok else "SPAWN_PROPAGATION_FAIL")
    raise SystemExit(0 if ok else 1)
