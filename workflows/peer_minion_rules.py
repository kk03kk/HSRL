"""Build the isolated three-minion sensitivity environment from an audited 37.0.2 tree."""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from tools.p2_data.check_engine_upstream import check, source_sha

RULES = "37.0.2-three-minion-sensitivity-v1"
BASE_HASH = "c4c71014e3d41760"
CARD_SHA = "7a02d10794cf80c56fa77b1dd235d49b892c8d3e31c2cf56887f696e2cf922a6"
CARDS = Path("engine/crates/hsbg-engine/data/cards.json")
CRATE = Path("engine/crates/hsbg-engine")


def replace_exact(path, old, new, count):
    value = path.read_text(encoding="utf8")
    if value.count(old) != count:
        raise ValueError(f"unexpected source at {path.name}")
    path.write_text(value.replace(old, new), encoding="utf8", newline="\n")


def prepare(out):
    import hsbg_engine
    source = source_identity(ROOT)
    if source["tracked_dirty"] or hsbg_engine.source_hash(ROOT/"engine") != BASE_HASH:
        raise ValueError("canonical source hash mismatch")
    if sha256((ROOT/CARDS).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != CARD_SHA:
        raise ValueError("canonical card table bytes differ")
    lock_path = ROOT/"config/engine-upstream-lock.json"
    check(ROOT, lock_path)
    out.mkdir(exist_ok=False)
    archive = subprocess.check_output(["git", "archive", "--format=zip", source["commit"], "engine"], cwd=ROOT)
    with ZipFile(io.BytesIO(archive)) as zipped:
        zipped.extractall(out)
    lock = json.loads(lock_path.read_bytes())
    # Only the normal card script is registered. Golden effects use the engine's own multiplier.
    content = out/CRATE/"src/content"
    replace_exact(content/"demons.rs", "m.counter += 1;\n                    if m.counter >= 3 {",
                  "m.counter += 1;\n                    if m.counter >= 4 {", 1)
    replace_exact(content/"quilboar.rs", "gem_bonus(c, me.side, 2 * me.g(), me.g())",
                  "gem_bonus(c, me.side, me.g(), me.g())", 2)
    for name in ("content/demons.rs", "content/quilboar.rs"):
        group = next(g for g in lock["groups"] if g["local"] == (CRATE/"src").as_posix())
        entry = group["files"][name]
        entry.update(fork_sha256=source_sha(out/CRATE/"src"/name), reason="Isolated three-minion sensitivity from the Nov 17 Blizzard hotfix.")
    atomic_json(out/"research-engine-lock.json", lock)
    check(out, out/"research-engine-lock.json")
    table = json.loads((out/CARDS).read_bytes())
    cards = {v["id"]:v for v in table["cards"]}
    for suffix in ("", "_G"):
        shark = cards["BG36_206"+suffix]
        assert shark["tier"] == 4
        shark["tier"] = 5
        champion = cards["BG23_017"+suffix]
        before, after = ("+4/+2","+2/+2") if suffix else ("+2/+1","+1/+1")
        for key in ("text","text_zh"):
            assert before in champion[key]
            champion[key] = champion[key].replace(before, after)
        boar = cards["BG28_633"+suffix]
        for key in ("text","text_zh"):
            assert boar[key].count("3") == 2
            boar[key] = boar[key].replace("3","4")
    table["meta"]["server_patch"] = RULES
    (out/CARDS).write_text(json.dumps(table, ensure_ascii=False, separators=(",",":"))+"\n",
                           encoding="utf8", newline="\n")
    # Execute new mechanic assertions in isolation, retaining the canonical tests in the parent.
    tests = out/CRATE/"tests"
    old = (tests/"s_tier_engines.rs").read_text(encoding="utf8")
    start = old.index("#[test]\nfn felboar_consumes_a_tavern_minion_after_three_spells")
    end = old.index("\n#[test]", start+8)
    test = old[start:end].replace("after_three_spells", "after_four_spells").replace("in [1, 2]", "in [1, 2, 3]")
    (tests/"three_minion_sensitivity.rs").write_text(old[:old.index("#[test]")]+test+"""
#[test]
fn champion_and_shark_use_sensitivity_rules() {
    for (card, scale) in [("BG23_017",1), ("BG23_017_G",2)] {
        let mut g = game(392034);
        set_board(&mut g, 0, &[]);
        give(&mut g, 0, card);
        play_last(&mut g, 0);
        assert_eq!((g.players[0].stats.gem_atk,g.players[0].stats.gem_hp), (scale,scale));
        kill(&mut g, 0);
        assert_eq!((g.players[0].stats.gem_atk,g.players[0].stats.gem_hp), (2*scale,2*scale));
    }
    for card in ["BG36_206", "BG36_206_G"] { assert_eq!(db().get(cid(card)).tier, 5); }
}
""", encoding="utf8", newline="\n")
    atomic_json(out/"identity.json", dict(source=source, canonical_source=BASE_HASH, source_hash=hsbg_engine.source_hash(out/"engine"),
        rules=RULES, card_table=file_identity(out/CARDS), canonical_card_sha= CARD_SHA, upstream_lock=file_identity(lock_path),
        research_lock=file_identity(out/"research-engine-lock.json"), rule_tests=file_identity(tests/"three_minion_sensitivity.rs"),
        scope="Felboar, Sanguine Champion and Snarky Shark; other rules remain those of the canonical parent."))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    prepare(parser.parse_args().out)
