"""Pure gesture budgets. No native APIs are acquired here."""
from dataclasses import dataclass
import math
from .contracts import Blocked


@dataclass(frozen=True)
class InputTiming:
    hover:float=.06
    hold:float=.06
    drag:float=.12
    release:float=.12
    double_gap:float=.09
    steps:int=12
    name:str='balanced-v2'

    def __post_init__(self):
        for name,minimum in (('hover',.04),('hold',.04),('drag',.096),('release',.10),('double_gap',.06)):
            value=getattr(self,name)
            if not isinstance(value,(float,int)) or not math.isfinite(value) or not minimum<=value<=1:
                raise Blocked('invalid input timing:'+name)
        if type(self.steps) is not int or not 8<=self.steps<=32:raise Blocked('invalid drag steps')


BALANCED=InputTiming()
CONSERVATIVE=InputTiming(hover=.08,drag=.24,release=.18,name='conservative-v2')


def timing_for(name):
    if name=='balanced':return BALANCED
    if name=='conservative':return CONSERVATIVE
    raise Blocked('unknown input timing profile:'+str(name))


def validate_plan(plan):
    """Validate every primitive before sending the first mouse-down."""
    import numbers
    if not isinstance(plan,(tuple,list)):raise Blocked('invalid input plan')
    for row in plan:
        if not isinstance(row,(tuple,list)) or not row:raise Blocked('invalid input primitive')
        kind=row[0]
        count=3 if kind=='drag' else 2
        if kind not in ('click','right_click','drag','double_click','move_pointer') or len(row)!=count:
            raise Blocked('invalid input primitive:'+str(kind))
        for point in row[1:]:
            if not isinstance(point,(tuple,list)) or len(point)!=2 or any(
                    isinstance(x,bool) or not isinstance(x,numbers.Real) or not math.isfinite(x) or not 0<x<1 for x in point):
                raise Blocked('point outside client or nonfinite')
