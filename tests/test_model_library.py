"""模型库清单自洽性；`artifacts/` 不入 Git，没有构建时跳过。"""
import json
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[1] / "artifacts" / "models" / "library-v1"


def manifest():
    p = LIB / "manifest.json"
    if not p.exists():
        pytest.skip("library-v1 not built (artifacts/ is local-only)")
    return json.loads(p.read_text(encoding="utf-8"))


def test_ids_unique_and_required_fields():
    m = manifest()
    ids = [e["id"] for e in m["models"]]
    assert len(ids) == len(set(ids))
    assert m["counts"]["weight_payloads"] == len(ids) == 67
    for e in m["models"]:
        for key in ("id", "family", "status", "league", "params", "arch",
                    "card_table", "compat", "weights_sha256", "file_sha256"):
            assert key in e, (e["id"], key)
        assert len(e["weights_sha256"]) == 64
        assert len(e["file_sha256"]) == 64


def test_linked_files_exist_and_are_covered_by_sums():
    m = manifest()
    sums = {}
    for line in (LIB / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, path = line.split("  ", 1)
        sums[path] = digest
    for e in m["models"]:
        if not e.get("library_model"):
            continue
        p = LIB / e["library_model"]
        assert p.exists(), e["id"]
        assert sums.get(e["library_model"]) == e["file_sha256"], e["id"]


def test_eval_ready_implies_load_and_forward():
    for e in manifest()["models"]:
        if e["eval_ready"]:
            assert e["compat"]["load_ok"] and e["compat"]["forward_ok"], e["id"]
            assert e["compat"]["forward_shape"] == [1, 245], e["id"]


def test_cohorts_are_subsets():
    m = manifest()
    ids = {e["id"] for e in m["models"]}
    c = json.loads((LIB / "cohorts" / "cohorts.json").read_text(encoding="utf-8"))
    for name, body in c["cohorts"].items():
        assert set(body["members"]) <= ids, name
        assert body["members"], name


def test_mainline_is_strict_league_a():
    m = manifest()
    by_id = {e["id"]: e for e in m["models"]}
    c = json.loads((LIB / "cohorts" / "cohorts.json").read_text(encoding="utf-8"))
    main = c["cohorts"]["mainline-v1"]["members"]
    assert len(main) == 8
    assert all(by_id[i]["league"] == "A-current-3702" for i in main)
    # 非主榜的 A 类只能是早期 tiny
    others = {e["id"] for e in m["models"]
              if e["league"] == "A-current-3702"} - set(main)
    assert all(by_id[i]["family"] == "az-tiny-0.96m" for i in others)
