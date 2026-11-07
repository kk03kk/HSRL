"""强制升本长训的 GPU 工程验收：复用 ``run_7357_preflight`` 并固定 ``--profile forced``。

单独成文件，是因为 NUS 启动器按「task → 脚本 + 默认输出目录」分派，且它只传 ``--out``；
这里在进入 ``main()`` 前注入 profile，避免启动器需要传额外参数。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from workflows.run_7357_preflight import main as preflight_main
    argv = sys.argv[1:]
    if '--profile' not in argv:
        argv = ['--profile', 'forced', *argv]
    sys.argv = [sys.argv[0], *argv]
    preflight_main()


if __name__ == '__main__':
    main()
