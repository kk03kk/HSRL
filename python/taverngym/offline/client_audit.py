"""客户端记录可用性报告；未知字段不伪装成已支持。"""
import json
from pathlib import Path


def snapshots(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        rows = value if isinstance(value, list) else [value]
    except ValueError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    for row in rows:
        value = row.get("sample", row)
        yield value.get("observation", value)


def audit(path, allowed_builds):
    from .bridge import ClientSnapshot, BridgeBlocked
    reports = []
    for n, data in enumerate(snapshots(path)):
        missing = [k for k in ("game_id", "revision", "build", "turn", "gold", "tavern_tier", "upgrade_cost")
                   if data.get(k) is None]
        try:
            snapshot = ClientSnapshot.from_dict(data)
            build_ok = snapshot.build in allowed_builds
            status = "needs_policy_context" if build_ok else "unsupported_build"
        except BridgeBlocked as exc:
            status = str(exc)
        reports.append({"record": n, "phase": data.get("phase"), "build": data.get("build"),
                        "missing_basic_fields": missing, "status": status,
                        "legacy_candidates_are_native_mask": False,
                        "context_present": isinstance(data.get("policy_context"), dict),
                        "requires_reconstruction": ["full global counters", "quest and pending semantics",
                                                    "public opponent history", "effect tokens", "native 245 legal bindings"],
                        "policy_input_accepted": False})
    return {"format": "taverngym.client-audit.v1", "source": str(path), "records": reports,
            "notice": "inventory only; use encode_snapshot for acceptance; no inference or OS input"}
