"""Build-bound numeric names from this user's static client DBF extraction."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from .contracts import Blocked

@lru_cache(maxsize=1)
def mapping(build):
    if str(build)!='255790':return {}
    p=Path(__file__).with_name('data')/'game_tags_255790.json'
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='c7e3b9b7ecaaac45bbc4f7e6b74f517c3c70e52291025bf43feb48b6827a9922':
        raise Blocked('client numeric tag dictionary changed')
    return json.loads(raw)


def symbolic_line(line,build):
    names=mapping(build)
    return re.sub(r'(?<=tag=)\d+(?= value=)',lambda m:names.get(m[0],m[0]),line) if names else line
