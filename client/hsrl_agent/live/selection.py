"""便携的模型选择；真实输入版本检查与离线推理明确分离。"""
import hashlib
import json
from pathlib import Path
from .contracts import Blocked

def selected_checkpoint(root):
    root=Path(root).resolve()
    config=json.loads((root/'config/live/model.json').read_text(encoding='utf8'))
    if config.get('schema')!='hsai.live-model/v1':raise Blocked('invalid selected model configuration')
    path=(root/config['checkpoint']).resolve()
    if not path.is_relative_to(root):raise Blocked('selected checkpoint must stay inside workspace')
    return path

def validate_live_profile(checkpoint,profile):
    """在创建窗口后端之前检查；不把旧版UI配置自动批准为新版。"""
    from .policy import POLICY_SPECS
    with Path(checkpoint).open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    spec=POLICY_SPECS.get(sha)
    if spec is None:raise Blocked('checkpoint is not an approved pinned policy')
    required=spec.get('required_client_build')
    # Build numbers are ints in POLICY_SPECS but strings in the reviewed profile JSON.
    # Comparing them raw made the gate unsatisfiable even for an identical build
    # (255790 in ['255790'] is False), so every autoplay start was refused.
    allowed={str(build) for build in profile.get('allowed_builds') or ()}
    if required is not None and (profile.get('review_state')!='reviewed' or str(required) not in allowed):
        raise Blocked(f'model requires a reviewed client profile for build {required}; inherited profile is historical')
    return spec
