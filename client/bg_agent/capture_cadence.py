"""Skip only passive screenshots; each actionable state still needs pixels."""
from .model import Phase


def passive_wait(previous,current,capture_end,now,interval=.35,*,processing=False):
    if previous is None or (current.game_id,current.phase,current.turn)!=(previous.game_id,previous.phase,previous.turn):return False
    if not 0<=now-capture_end<interval:return False
    if current.interaction is not None or current.choices:return False
    if current.phase==Phase.COMBAT:return True
    if current.phase==Phase.RECRUIT and current.options_pending:return True
    # The CURRENT task/root is authoritative even on the first post-input
    # cycle. A pre-input frame has no task ID to match. This only postpones
    # pixels while input is forbidden; it never grants readiness.
    return bool(current.phase==Phase.RECRUIT and (processing or current.client_task_list is not None))
