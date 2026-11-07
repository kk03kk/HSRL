#!/usr/bin/env python3
"""Verify a HSAI model-library manifest against the files on disk.

Checks, per model:
  1. hardlinked/present model.pt (and trainer.pt) match the declared file SHA256
  2. weights_sha256 (canonical tensor content) matches the manifest
  3. hsbg.az can load the checkpoint (for A/B leagues)
  4. a real 3227+124 -> 245 forward is finite
  5. declared card-table status matches the current 37.0.2 tables

Usage:
    python tools/verify_model_library.py
    python tools/verify_model_library.py --cohort mainline-v1
    python tools/verify_model_library.py --models entity16m-nus-7357 --quick
    python tools/verify_model_library.py --json artifacts/local/library-verify.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "artifacts" / "models" / "library-v1"
sys.path.insert(0, str(ROOT))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def canon_hash(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        if not hasattr(v, "shape"):
            continue
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def load_state(path: Path) -> dict:
    pay = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(pay, dict):
        for key in ("state_dict", "model"):
            if isinstance(pay.get(key), dict):
                return pay[key]
        if pay and all(hasattr(v, "shape") for v in pay.values()):
            return pay
    raise ValueError("no state dict in payload")


def check_model(entry: dict, quick: bool) -> dict:
    res = {"id": entry["id"], "ok": True, "checks": {}, "errors": []}

    def fail(msg):
        res["ok"] = False
        res["errors"].append(msg)

    if not entry.get("library_model"):
        if entry["status"] in ("archive-only", "test-fixture"):
            res["checks"]["skipped"] = "not linked (archive/test)"
            return res
        fail("manifest has no library_model")
        return res

    path = LIB / entry["library_model"]
    if not path.exists():
        fail(f"missing {path}")
        return res
    h = sha256_file(path)
    res["checks"]["file_sha256"] = h == entry["file_sha256"]
    if h != entry["file_sha256"]:
        fail(f"file sha mismatch {h}")

    if quick:
        return res

    try:
        sd = load_state(path)
        res["checks"]["weights_sha256"] = canon_hash(sd) == entry["weights_sha256"]
        if not res["checks"]["weights_sha256"]:
            fail("canonical weights hash mismatch")
    except Exception as exc:  # noqa: BLE001
        fail(f"state load: {type(exc).__name__}: {exc}")
        return res

    if not entry["eval_ready"]:
        res["checks"]["skipped"] = "declared not eval_ready"
        return res

    try:
        from hsbg.az.train.model import load_checkpoint
        from hsbg.az.train.entity_model import load_card_tables
        model, _meta = load_checkpoint(str(path), device="cpu")
        model.eval()
        with torch.no_grad():
            f = torch.zeros(1, 3227)
            c = torch.zeros(1, 124, dtype=torch.long)
            mask = torch.ones(1, 245, dtype=torch.bool)
            out = model(f, c, mask) if getattr(model, "uses_cards", False) else model(f, c)
        res["checks"]["forward_245_finite"] = bool(
            out[0].shape == (1, 245) and torch.isfinite(out[0]).all())
        if not res["checks"]["forward_245_finite"]:
            fail("forward shape/finite check failed")
        declared = entry["card_table"]["status"]
        cs = getattr(model, "card_static", None)
        if declared in ("current-3702", "legacy-pre3702") and cs is not None:
            tables = load_card_tables()
            same = bool(np.array_equal(cs.detach().cpu().numpy(), tables.static))
            res["checks"]["card_table_status"] = (same if declared == "current-3702"
                                                 else not same)
            if not res["checks"]["card_table_status"]:
                fail(f"declared card table {declared} but exact={same}")
    except Exception as exc:  # noqa: BLE001
        fail(f"load/forward: {type(exc).__name__}: {exc}")

    tr = entry.get("trainer") or {}
    if tr.get("path") and (LIB / "models" / entry["id"] / "trainer.pt").exists():
        tp = LIB / "models" / entry["id"] / "trainer.pt"
        res["checks"]["trainer_sha256"] = sha256_file(tp) == tr["sha256"]
        if not res["checks"]["trainer_sha256"]:
            fail("trainer sha mismatch")
    return res


def main() -> int:
    global LIB
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", default=str(LIB), help="library root (contains manifest.json)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--models", default=None, help="comma separated ids")
    ap.add_argument("--quick", action="store_true", help="hash only, no torch load")
    ap.add_argument("--json", default=None, help="write full result json here")
    args = ap.parse_args()

    LIB = Path(args.lib).resolve()
    manifest_path = Path(args.manifest) if args.manifest else LIB / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    entries = {e["id"]: e for e in manifest["models"]}
    want = None
    if args.cohort:
        cohorts = json.loads((LIB / "cohorts" / "cohorts.json").read_text(encoding="utf-8"))
        want = cohorts["cohorts"][args.cohort]["members"]
    if args.models:
        want = [x.strip() for x in args.models.split(",") if x.strip()]

    selected = [entries[i] for i in want] if want else list(manifest["models"])
    results = []
    for e in selected:
        r = check_model(e, args.quick)
        results.append(r)
        flag = "OK  " if r["ok"] else "FAIL"
        print(f"[{flag}] {e['id']:38s} {r['checks']}", flush=True)

    bad = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
        print("wrote", args.json)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
