"""Named, bounded timings; experimental speeds require real-client validation."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class InteractionTiming:
    name: str='safe'
    hover: float=.06
    drag_step: float=.008
    click_hold: float=.07
    release_hold: float=.10
    drag_steps: int=12
    region_quiet: float=.25
    log_settle: float=.15

    def __post_init__(self):
        for key in ('hover','drag_step','click_hold','release_hold','region_quiet','log_settle'):
            value=getattr(self,key)
            if not math.isfinite(value) or value<=0 or value>1:
                raise ValueError('invalid_interaction_timing:'+key)
        if not isinstance(self.drag_steps,int) or not 4<=self.drag_steps<=32:
            raise ValueError('invalid_drag_steps')


SAFE_TIMING=InteractionTiming()
EXPERIMENTAL_TIMING=InteractionTiming(name='experimental',hover=.04,drag_step=.006,
                                    release_hold=.08,region_quiet=.20)


def timing_profile(name):
    if name=='safe':return SAFE_TIMING
    if name=='experimental':return EXPERIMENTAL_TIMING
    raise ValueError('unknown_interaction_timing:'+str(name))
