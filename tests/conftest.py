"""测试前置：只保证仓库根在 ``sys.path``。

AutoTavern 引擎（``hsbg_engine``）未构建时，依赖它的测试自行 ``importorskip``。
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 客户端为独立可选安装；离线接口测试直接使用本工作区源码。
sys.path.insert(0, str(ROOT / "client"))


@pytest.fixture
def inductor_compiler():
    """只对实际编译的集成用例检查本机 C++ 工具链，eager/回退用例照常测试。"""
    compiler = "cl" if sys.platform == "win32" else "c++"
    if shutil.which(compiler) is None:
        pytest.skip(f"torch Inductor 集成测试需要 {compiler} 编译器")
