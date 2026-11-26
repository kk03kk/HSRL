import json

import taverngym


def test_versions_match_card_data():
    card_data = taverngym.REPO_ROOT / "crates" / "taverngym" / "data" / "cards.json"
    meta = json.loads(card_data.read_text(encoding="utf-8"))["meta"]
    assert taverngym.game_version() == meta["client_version"] == "37.0.2.256589"
    assert taverngym.server_rules_version() == meta["server_patch"] == "37.0.3"
    assert taverngym.version()


def test_the_extension_records_the_sources_it_was_built_from():
    # The card data does not change when the engine's code does: the source hash tells a
    # stale binary from a current one.
    info = taverngym.build_info()
    assert info["file"] and info["profile"] in ("release", "debug")
    assert len(info["source_hash"]) == 16
    # Built from this checkout before the tests run.
    assert info["source_hash"] == taverngym.source_hash()
    assert taverngym.build_is_current() is True
    # A copied python/ tree has no sources to compare with.
    assert taverngym.source_hash("/nonexistent") is None
    assert taverngym.build_is_current("/nonexistent") is None


def test_runtime_card_data_matches_real_hotfix_observations():
    fixtures = taverngym.REPO_ROOT / "crates" / "taverngym" / "tests" / "fixtures"
    evidence = json.loads((fixtures / "live_3703_hotfix.json").read_text(encoding="utf-8"))
    assert taverngym.server_rules_version() == evidence["server_patch_window"]
    catalog = {card["id"]: card for card in json.loads(taverngym.catalog_json())}
    for case in evidence["observations"]:
        assert catalog[case["id"]][case["field"]] == case["value"], case
