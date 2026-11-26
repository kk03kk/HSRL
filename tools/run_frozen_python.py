"""Run repository Python code against the verified extension installed in the run environment."""
import os
from pathlib import Path
import runpy
import sys

import taverngym

root = Path(__file__).resolve().parents[1]
if taverngym.build_is_current(root) is not True:
    raise SystemExit("The installed simulator extension differs from this source checkout")
if expected_gpu := os.environ.get("BC_EXPECTED_GPU"):
    import torch
    actual_gpu = torch.cuda.get_device_name()
    if actual_gpu != expected_gpu:
        raise SystemExit(f"Expected {expected_gpu}, got {actual_gpu}")
taverngym.__path__.insert(0, str(root / "python" / "taverngym"))
del sys.argv[0]
runpy.run_path(sys.argv[0], run_name="__main__")
