"""Speculative, read-only inference. It never authorizes client input.

Policies opt in with a thread-safe, side-effect-free preview(observation).
Only an exact semantic/candidate match may be adopted on the controller thread.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import threading
import time


def policy_input(obs):
    """Logical policy view; full causal snapshots belong to the verifier.

    Receipt identities/transitions remain available, but copying sixteen full
    historical boards on every inference/plan serves no policy requirement.
    Never mutate the live observation or durable execution checkpoint.
    """
    return replace(obs,interaction=replace(obs.interaction,display_state='unknown') if obs.interaction else None,
        resolutions=tuple(replace(r,post_state=None) if r.post_state is not None else r for r in obs.resolutions))


def policy_snapshot(obs):
    """Remove acquisition/UI timing, keeping all policy-relevant state/actions."""
    return replace(policy_input(obs),revision=0,captured_at=0,source_updated_at=None,log_event_at=None,
                   log_position=None)


def snapshot_key(obs):
    return hashlib.sha256(json.dumps(policy_snapshot(obs).to_dict(),sort_keys=True,
                                    separators=(',',':')).encode()).hexdigest()


#: 决策输入之外、只会造成“假变化”的字段：
#: 版本/采集时间、日志游标、客户端提交/回放游标、以及历史回执（resolutions）。
#: 它们让同一局面的 snapshot_key 每帧都不同，导致预判缓存/decision-ahead 白算，
#: 也让投递前的复核把无关日志噪声当成“状态变了”而取消动作。
_UNSTABLE_DECISION_FIELDS = ('captured_at', 'source_updated_at', 'log_event_at', 'revision',
                             'log_position', 'client_submission_position',
                             'client_playback_position', 'client_task_list', 'resolutions')


def decision_key(obs):
    """只覆盖“策略会读到的输入”的稳定键；用于预判采用、缓存与投递前复核。"""
    data = policy_snapshot(obs).to_dict()
    for key in _UNSTABLE_DECISION_FIELDS:
        data.pop(key, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class PolicyPreview:
    def __init__(self,preview,observation):
        # Frozen dataclasses still contain mutable tag dictionaries.
        self.observation=deepcopy(policy_input(observation))
        self.key=decision_key(self.observation)
        self.done=threading.Event();self.decision=None;self.error=None;self.elapsed_ms=None
        self.started=time.monotonic()
        def run():
            started=time.perf_counter()
            try:self.decision=preview(self.observation)
            except Exception as exc:self.error=type(exc).__name__+': '+str(exc)
            finally:
                self.elapsed_ms=(time.perf_counter()-started)*1000
                self.done.set()
        self.thread=threading.Thread(target=run,name='policy-preview',daemon=True)
        self.thread.start()
