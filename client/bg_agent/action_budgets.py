"""等待上限只用于失败界定，不用于决定动作就绪。"""
from .model import Kind

CAPTURE_MAX_AGE=.75
PREPARE_SECONDS=4
CHOICE_SECONDS=45
TARGET_SECONDS=8
CONFIRM_SECONDS=8
QUEUE_DISMISS_SECONDS=8
# 历史全屏捕获中位约57ms、P90约67ms；为映射和信号分析留出余量。
# 这是捕获起点之间的最小间隔，不在处理结束后叠加固定等待。
CONTROL_POLL_SECONDS=.12
CONTROL_WAIT_SECONDS=240
BUY_RECOVERY_SECONDS=8
RESULT_SECONDS={Kind.DISMISS:12,Kind.CHOOSE:CHOICE_SECONDS,Kind.PLAY:45,
    Kind.HERO_POWER:45,Kind.ACTIVATE:45,Kind.CONFIRM:CHOICE_SECONDS,
    Kind.TARGET:TARGET_SECONDS,Kind.SELECT:8}


def result_timeout(action,session_seconds=2400):
    return session_seconds if action.kind in (Kind.ENTER,Kind.END) else RESULT_SECONDS.get(action.kind,5)


def prepare_timeout(action,phase):
    if str(phase)=='choice' and action.kind in (Kind.CHOOSE,Kind.SELECT,Kind.CONFIRM):return CHOICE_SECONDS
    if action.kind==Kind.TARGET:return TARGET_SECONDS
    if action.kind==Kind.CONFIRM:return CONFIRM_SECONDS
    return PREPARE_SECONDS
