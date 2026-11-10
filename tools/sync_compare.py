"""Compare two ``sync_manifest`` files and print the drift report.

The two manifests are produced by the same script on different machines; the
diff *is* the cross-side drift check (D-003, D-014). Rules:

* pins are compared by **Git blob** when both sides have one (CRLF-proof);
  otherwise by worktree SHA256 with an explicit "raw-diff" note;
* a pin whose ``scope`` is ``local``/``server`` is allowed to be absent on the
  other side — that is a documented division of truth, not drift;
* trees are compared by file count/bytes with a tolerance, because the server
  legitimately keeps generated files under ``engine/`` and ``records/``;
* artifacts/records are compared by *presence and size*, never by bytes.

Exit code: 0 = no hard drift, 1 = at least one hard drift.

Usage
-----
    python tools/sync_compare.py docs/sync/manifests/local.json docs/sync/manifests/server.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TREE_TOLERANCE = 0.35      # 35% file-count difference is tolerated per tree
ARTIFACT_TOLERANCE = 0.10  # 10% size difference is tolerated per artifact root


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pin_value(entry: dict | None, *, prefer_blob: bool) -> tuple[str | None, str]:
    """Use the same identity on both sides, or the comparison is meaningless.

    ``git_blob`` is a SHA1 blob id, ``sha256`` is a raw hash; mixing them would
    report every pin as different. Blob ids are only comparable when *both*
    sides provide one (i.e. both ran the v2 script).
    """
    if not entry:
        return None, "absent"
    if prefer_blob and entry.get("git_blob"):
        return entry["git_blob"], "git-blob"
    if entry.get("sha256"):
        return entry["sha256"], "worktree-sha256"
    if entry.get("git_blob"):
        return entry["git_blob"], "git-blob"
    return None, "absent"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("left", type=Path)
    ap.add_argument("right", type=Path)
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    args = ap.parse_args(argv)
    left, right = load(args.left), load(args.right)

    hard: list[str] = []
    soft: list[str] = []
    lines: list[str] = []
    report: dict = {"left": str(args.left), "right": str(args.right), "pins": [], "trees": [], "artifacts": []}

    lg, rg = left.get("git", {}), right.get("git", {})
    lines.append("git: %s %s (dirty=%s)  vs  %s %s (dirty=%s)" % (
        left.get("side"), (lg.get("head") or "?")[:10], lg.get("dirty"),
        right.get("side"), (rg.get("head") or "?")[:10], rg.get("dirty")))
    if lg.get("head") and rg.get("head") and lg["head"] != rg["head"]:
        soft.append("different HEAD (%s vs %s) — pull/rebase before comparing again"
                    % (lg["head"][:10], rg["head"][:10]))

    lkeys = set(left.get("pins", {}))
    rkeys = set(right.get("pins", {}))
    both_blob = all((left["pins"].get(k) or {}).get("git_blob") and (right["pins"].get(k) or {}).get("git_blob")
                    for k in lkeys & rkeys) if (lkeys & rkeys) else False
    for key in sorted(lkeys | rkeys):
        le, re_ = left["pins"].get(key), right["pins"].get(key)
        scope = (le or re_ or {}).get("scope", "both")
        lv, lsrc = pin_value(le, prefer_blob=both_blob)
        rv, rsrc = pin_value(re_, prefer_blob=both_blob)
        same = lv == rv
        if lv is None and rv is None:
            tag = "ABSENT"
            soft.append("pin %s absent on both sides (archived) — drop it from PINS if permanent" % key)
        elif same:
            tag = "OK"
        else:
            missing_side = None
            if lv is None:
                missing_side = left.get("side")
            if rv is None:
                missing_side = right.get("side")
            if missing_side and scope in ("local", "server") and missing_side != scope:
                tag = "OK(scope)"
                soft.append("pin %s absent on %s, allowed by scope=%s" % (key, missing_side, scope))
            elif missing_side:
                tag = "MISSING(%s)" % missing_side
                hard.append("pin %s missing on %s (scope=%s)" % (key, missing_side, scope))
            else:
                hard.append("pin %s differs (%s vs %s; compared by %s/%s)" % (key, lv, rv, lsrc, rsrc))
                tag = "DIFF"
        lines.append("  %-12s %-58s %s %s" % (tag, key, (lv or "-")[:12], (rv or "-")[:12]))
        report["pins"].append(dict(path=key, scope=scope, left=lv, right=rv, same=same, tag=tag))

    for tree in sorted(set(left.get("trees", {})) | set(right.get("trees", {}))):
        lt, rt = left["trees"].get(tree, {}), right["trees"].get(tree, {})
        lf, rf = lt.get("files"), rt.get("files")
        if not lf or not rf:
            continue
        drift = abs(lf - rf) / max(lf, rf)
        tag = "OK" if drift <= TREE_TOLERANCE else "DIFF"
        if tag == "DIFF":
            hard.append("tree %s file count %s vs %s (%.0f%%)" % (tree, lf, rf, drift * 100))
        lines.append("  %-12s tree %-12s files %5s vs %5s   bytes %10s vs %10s" % (
            tag, tree, lf, rf, lt.get("bytes"), rt.get("bytes")))
        report["trees"].append(dict(tree=tree, left=lf, right=rf, drift=round(drift, 3), tag=tag))

    for root_name in sorted(set(left.get("artifacts", {})) | set(right.get("artifacts", {}))):
        la, ra = left["artifacts"].get(root_name, {}), right["artifacts"].get(root_name, {})
        lb, rb = la.get("bytes", 0) or 0, ra.get("bytes", 0) or 0
        if not la.get("exists") and not ra.get("exists"):
            continue
        drift = abs(lb - rb) / max(lb, rb, 1)
        tag = "OK" if drift <= ARTIFACT_TOLERANCE else "INFO"
        lines.append("  %-12s artifacts/%-10s %6.0f MB vs %6.0f MB   (%s)" % (
            tag, root_name, lb / 1e6, rb / 1e6, "same order" if tag == "OK" else "side-specific"))
        report["artifacts"].append(dict(name=root_name, left_bytes=lb, right_bytes=rb, tag=tag))

    for problem in left.get("problems", []):
        if not problem["exists"]:
            hard.append("local missing %s (%s)" % (problem["check"], problem["path"]))
    for problem in right.get("problems", []):
        if not problem["exists"]:
            soft.append("server missing %s (%s)" % (problem["check"], problem["path"]))

    report["hard"] = hard
    report["soft"] = soft
    print("\n".join(lines))
    if soft:
        print("\nsoft:")
        for item in soft:
            print("  -", item)
    if hard:
        print("\nhard drift:")
        for item in hard:
            print("  -", item)
    print("\nresult:", "DRIFT" if hard else "consistent (with documented exceptions)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
