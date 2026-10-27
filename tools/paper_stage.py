"""论文启发实验入口；Windows本地DLL环境沿用离线启动器。"""
import runpy
import sys
from local_offline import prepare

if __name__ == "__main__":
    handles = prepare()
    sys.argv = ["taverngym.research", *sys.argv[1:]]
    runpy.run_module("taverngym.research", run_name="__main__")
