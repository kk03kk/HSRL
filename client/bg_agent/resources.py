"""Runtime assets live with the package, independently of development archives."""
from pathlib import Path


def calibration_root():
    return Path(__file__).resolve().parent/'data'/'calibration'
