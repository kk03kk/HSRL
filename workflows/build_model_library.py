"""Build the standardized HSAI Battlegrounds model library (v1).

Inputs (read-only evidence produced by the inventory pass):
  artifacts/local/inventory/weight-scan.json
  artifacts/local/inventory/weights-sha256.json
  artifacts/local/inventory/canonical-hashes.json
  artifacts/local/inventory/verify-az.json
  artifacts/local/inventory/lineage.json
  records/research/long1085131/eval_rebuilt.json
  records/research/long1080772/paired7357-to7616.json
  records/nus-audits/2026-11-06/*-eval-summary.json

Output:
  artifacts/models/library-v1/manifest.json
  artifacts/models/library-v1/models/<id>/{model.pt,trainer.pt,model.json}
  artifacts/models/library-v1/cohorts/*.json
  artifacts/models/library-v1/SHA256SUMS.txt
  artifacts/models/library-v1/index.csv
  artifacts/models/library-v1/gaps.json

The builder never edits or deletes source weights: model files are **hardlinks**
to the canonical original, so the library adds directory entries only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(r"D:\HSAI")
INV = ROOT / "artifacts" / "local" / "inventory"
LIB = ROOT / "artifacts" / "models" / "library-v1"


def jread(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


# ── curated knowledge ────────────────────────────────────────────────────
# canon[0:12] -> identity / provenance
SPEC = {
    "7dad8115e7d1": dict(
        id="entity16m-s14-best", name="s14 Best (pre-3702 card table)",
        status="reference", run="s14", version=None, cum_decisions=None,
        note="原 s14 主模型；内嵌旧卡表（cost 约定 3）。与 publish-representative 权重逐位相同。"),
    "47137fed3ffe": dict(
        id="entity16m-s14-3702", name="s14 Best migrated to 37.0.2",
        status="reference", run="s14", version=None, cum_decisions=None,
        note="仅替换 card_static/card_text 缓冲区的 3702 迁移副本；学习参数与 s14-best 完全相同。"
             "它也是 1080772/1085131 全部 Trainer 的 anchor。"),
    "e872d32a1a71": dict(
        id="entity16m-nus-4159", name="NUS 1080772 initial (4159, 37.0.2 migration)",
        status="candidate", run="nus-1080772", version=4159, cum_decisions=198819917,
        note="由原版 4159（255790）迁移卡表而来，不是重新训练；原始 4159 文件（c8e…）不在本地。"),
    "5698588162fa": dict(
        id="entity16m-nus-5506", name="NUS 1080772 step 5506 (+150M)",
        status="candidate", run="nus-1080772", version=5506, cum_decisions=348831055,
        note=""),
    "f9f20633ecee": dict(
        id="entity16m-nus-6039", name="NUS 1080772 step 6039 (+210M)",
        status="production", run="nus-1080772", version=6039, cum_decisions=408863650,
        note="实机 245 桥当前白名单模型（config/live/model.json）；未在真实游戏验收。"),
    "9c7539263c58": dict(
        id="entity16m-nus-7357", name="NUS 1080772 step 7357 (approved resume point)",
        status="approved-resume", run="nus-1080772", version=7357, cum_decisions=558873732,
        note="AGENTS.md 批准的下一轮完整 Trainer 起点；Trainer 的 anchor 仍是 s14-3702，"
             "续训必须 reset-anchor=1。"),
    "a0f03edc4db9": dict(
        id="entity16m-nus-7710", name="NUS 1080772 step 7710 (user-paused end)",
        status="candidate", run="nus-1080772", version=7710, cum_decisions=599740461,
        note="安全暂停时的最后完整状态；没有已下载的棋力评测。"),
    "64881f4eb806": dict(
        id="entity16m-p50-8255-648m", name="P50 1085131 milestone 648.9M (v8255)",
        status="candidate", run="p50-7357-long-001", version=8255, cum_decisions=648873732,
        note="1085131 唯一下载了权重的中间里程碑（另有 7 个里程碑仅有评测与 SHA）；Trainer 未下载。"),
    "10fcd555198b": dict(
        id="entity16m-p50-10004-820m", name="P50 1085131 final step 10004 (820.3M)",
        status="candidate", run="p50-7357-long-001", version=10004, cum_decisions=820314909,
        note="用户安全暂停时的最终完整状态；只有 health 报告，没有 96 局棋力评测。"),
    "837ce92dc367": dict(
        id="entity16m-forced-7358-lockstep", name="forced-tier local smoke 7358 (lockstep)",
        status="local-smoke", run="forced-tier-smoke", version=7358, cum_decisions=None,
        note="本地工程冒烟（lockstep），不是棋力候选；anchor 已正确重置为 7357。"),
    "61e1e7e24ce4": dict(
        id="entity16m-forced-7358-serial", name="forced-tier local smoke 7358 (serial)",
        status="local-smoke", run="forced-tier-smoke", version=7358, cum_decisions=None,
        note="本地工程冒烟（serial）；与 smoke2 同权重。"),
    "0304ceff57a7": dict(
        id="entity16m-forced-7359-serial", name="forced-tier local smoke 7359 (serial)",
        status="local-smoke", run="forced-tier-smoke", version=7359, cum_decisions=None,
        note="本地工程冒烟第二轮；探针，不作为训练起点。"),
    "3bc9d21c1e97": dict(
        id="entity16m-test-7357-update-restore", name="7357 synthetic update/restore fixture",
        status="test-fixture", run="local-test", version=7358, cum_decisions=None,
        note="真实网络合成一次更新后的测试产物（artifacts/local/7357-update-restore），不是训练模型。"),
    "35a026f7e79a": dict(
        id="entity14m9-champion-c0-1975", name="champion c0 1975 (seq head)",
        status="historical", run="champion-c0-1975", version=None, cum_decisions=None,
        note="14.8M 骨干 + seq_head（14.90M）；旧卡表。"),
    "d7d4aec83347": dict(
        id="entity14m-original", name="original 14.8M (BC base, trainer_updates 201)",
        status="historical", run="autotavern", version=201, cum_decisions=None,
        note="所有 14.8M/16.4M entity 的 BC 共同祖先；内嵌旧卡表。"),
    "903fedf662d4": dict(
        id="entity14m-baseline5m", name="baseline 5M",
        status="historical", run="bot50m-delivery", version=None, cum_decisions=None, note=""),
    "f17a3353c961": dict(
        id="entity14m-hsbrsim-resume", name="hsbrsim train-resume (step 3)",
        status="historical", run="hsbrsim-train-resume", version=3, cum_decisions=None, note=""),
    "c5d1736e6ba1": dict(
        id="entity14m-friend-ckpt-5023397", name="friend eval ckpt 5,023,397",
        status="historical", run="full-training-audit", version=None, cum_decisions=None, note=""),
    "190ca60383ed": dict(
        id="entity14m-audit-bf16", name="training-audit bf16",
        status="historical", run="nus-audit-1031063", version=None, cum_decisions=None, note=""),
    "ff8d8eb5bd86": dict(
        id="entity14m-audit-fp32", name="training-audit fp32",
        status="historical", run="nus-audit-1031063", version=None, cum_decisions=None, note=""),
    "94a6325e46f8": dict(
        id="entity14m-bot50m-provisional-best", name="bot50m provisional bot best",
        status="historical", run="nus-bot50m-v1", version=None, cum_decisions=None, note=""),
    "4f766e03fe46": dict(
        id="entity14m-champion-policy", name="champion c0 1975 policy export",
        status="historical", run="champion-c0-1975", version=None, cum_decisions=None, note=""),
}

TINY_SPEC = {
    "99e2c623c8c4": ("tiny-gpu1079842-seed", "gpu1079842 az-seeded"),
    "fa346628cfa1": ("tiny-gpu1079842-exact", "gpu1079842 az-seeded-exact (= repeat)"),
    "55451e9885e8": ("tiny-opponent-controller-final", "opponent-controller final (step 16)"),
    "0ded88206c55": ("tiny-opponent-controller-step15", "opponent-controller step 15"),
    "078a382d81d5": ("tiny-scaling-final", "scaling-final (step 26)"),
    "4994da1b6bb2": ("tiny-scaling-step24", "scaling-final step 24"),
    "3a7de8cdcfbd": ("tiny-scaling-step25", "scaling-final step 25"),
}
SPEC.update({k: dict(id=v[0], name=v[1], status="historical", run="tiny-era",
                     version=None, cum_decisions=None,
                     note="3227→256→245 的早期 MLP；不编码 cards。")
             for k, v in TINY_SPEC.items()})

# protocol / library identifiers
P_1080772 = dict(
    protocol_sha256="e51924edb562fb65a4e5f1b3511f658716b7e39ddf1e02aee46fb909fcc3f072",
    library="bot-firestone-3702-economy-v2-live-v1/dev", games=96, seeds=12)
P_P50 = dict(
    protocol_sha256="8a02ceb4e9ed4ac1c0c8e6e029d3defac07999ee614536b7098c49f2a0c9211f",
    library="bot-p50-3702-v1/dev (d493ef78…)", games=96, seeds=12)

EVAL = {
    "e872d32a1a71": [dict(**P_1080772, mean_rank=4.385416666666667, top4_rate=0.625,
                          win_rate=0.010416666666666666,
                          source="records/nus-audits/2026-11-06/initial-4159-3702-eval-summary.json")],
    "5698588162fa": [dict(**P_1080772, mean_rank=3.78125, top4_rate=0.8125,
                          win_rate=0.041666666666666664,
                          source="records/nus-audits/2026-11-06/step-005506-150m-eval-summary.json")],
    "f9f20633ecee": [dict(**P_1080772, mean_rank=3.8333333333333335, top4_rate=0.7083333333333334,
                          win_rate=0.052083333333333336,
                          source="records/nus-audits/2026-11-06/step-006039-210m-eval-summary.json")],
    "9c7539263c58": [
        dict(**P_1080772, mean_rank=3.6666666666666665, top4_rate=None, win_rate=None,
             source="records/research/long1080772/paired7357-to7616.json (before)"),
        dict(**P_P50, mean_rank=4.46875, top4_rate=0.59375, win_rate=0.010416666666666666,
             source="records/research/long1085131/eval_rebuilt.json (baseline)")],
    "64881f4eb806": [dict(**P_P50, mean_rank=4.270833333333333, top4_rate=0.5833333333333334,
                          win_rate=0.010416666666666666,
                          delta_vs_7357=-0.19791666666666666,
                          ci95=[-0.6145833333333334, 0.2708333333333333],
                          source="records/research/long1085131/eval_rebuilt.json")],
}

# canon12 -> (parent canon12, l2, rel, evidence)
LINEAGE = {
    "47137fed3ffe": ("7dad8115e7d1", 0.0, 0.0,
                     "学习参数逐位相同；仅 card_static/card_text 被替换为 3702 卡表"),
    "e872d32a1a71": ("47137fed3ffe", 5.243, 0.0113, "198.8M 累计决策续训"),
    "5698588162fa": ("e872d32a1a71", 3.668, 0.0079, "+150.0M 决策"),
    "f9f20633ecee": ("5698588162fa", 1.808, 0.0039, "+60.0M 决策"),
    "9c7539263c58": ("f9f20633ecee", 2.889, 0.0062, "+150.0M 决策"),
    "a0f03edc4db9": ("9c7539263c58", 1.487, 0.0032, "+40.9M 决策"),
    "64881f4eb806": ("9c7539263c58", 2.357, 0.0051, "P50 分叉，+90.0M 决策"),
    "10fcd555198b": ("64881f4eb806", 2.717, 0.0058, "+171.4M 决策"),
    "837ce92dc367": ("9c7539263c58", 0.020, 0.0000, "forced-tier 本地冒烟 1 轮"),
    "61e1e7e24ce4": ("9c7539263c58", 0.019, 0.0000, "forced-tier 本地冒烟 1 轮"),
    "0304ceff57a7": ("61e1e7e24ce4", 0.031, 0.0001, "forced-tier 本地冒烟 2 轮"),
    "3bc9d21c1e97": ("9c7539263c58", 0.002, 0.0000, "合成一次更新（测试夹具）"),
}

GAPS = [
    dict(id="entity16m-nus-7616", version=7616, cum_decisions=None,
         weights_sha256="2b5ac868ed82ae1300dd764d21afdb4f869364d5f4c18c3820bfe4e8a223ecd1",
         detail="1080772 的 7616 只有配对评测（vs 7357 +0.156，CI[-0.209,+0.500]），权重未下载。",
         action="只在 NUS 旧 run 目录仍保留时按 SHA 补取；不要用 7710 冒充。"),
    dict(id="entity16m-p50-7638-588m", version=7638, cum_decisions=588873732,
         weights_sha256="dab63ec05b2c7dc01678563f7c61f2785c49fe647cc3e9f8d7ad1edef31ecb82",
         detail="1085131 里程碑，已有 96 局评测（4.583/56.3%/0.0%），本地无权重。",
         action="从 NUS 原 run 的 milestones/decisions-588873732 按 SHA 补取。"),
    dict(id="entity16m-p50-7945-618m", version=7945, cum_decisions=618873732,
         weights_sha256="9e9cc801f2880e04de9be46002b259dbfc172d8bbb0abcd4f1be4c9110547944",
         detail="1085131 里程碑，已有 96 局评测（4.615/52.1%/1.0%），本地无权重。",
         action="同上（decisions-618873732）。"),
    dict(id="entity16m-p50-8561-678m", version=8561, cum_decisions=678873732,
         weights_sha256="87ea2735192e64e5a909ed554ab6d972c0759174a610178e9ead5ab4a20dfdcb",
         detail="1085131 里程碑，已有 96 局评测（4.531/54.2%/2.1%），本地无权重。",
         action="同上（decisions-678873732）。"),
    dict(id="entity16m-p50-8865-708m", version=8865, cum_decisions=708873732,
         weights_sha256="4eddf5bf62b98aec98639ecdeea51739059a14359b6f398fae3f35e513ed45e2",
         detail="1085131 里程碑，已有 96 局评测（4.708/49.0%/0.0%），本地无权重。",
         action="同上（decisions-708873732）。"),
    dict(id="entity16m-p50-9171-738m", version=9171, cum_decisions=738873732,
         weights_sha256="aa95cbd3685c35112af64838d606fd6263c925233824d6f7294f159ce37c5a54",
         detail="1085131 里程碑，已有 96 局评测（4.542/55.2%/3.1%），本地无权重。",
         action="同上（decisions-738873732）。"),
    dict(id="entity16m-p50-9479-768m", version=9479, cum_decisions=768873732,
         weights_sha256="5a1af644f129058c121cbb8a5f057993a978739c51968c811fdfe708e3e69f1a",
         detail="1085131 里程碑，已有 96 局评测（4.427/57.3%/0.0%），本地无权重。",
         action="同上（decisions-768873732）。"),
    dict(id="entity16m-p50-9786-798m", version=9786, cum_decisions=798873732,
         weights_sha256="988fdc7fedb8fc8d794b5db00914e58a69754900c3a3d7bc55e2745d608bd2a3",
         detail="1085131 最后一个被评测的里程碑（4.490/54.2%/1.0%），本地无权重。",
         action="同上（decisions-798873732）。"),
    dict(id="trainer-p50-8255-648m", version=8255, cum_decisions=648873732,
         weights_sha256="6c0a8a34276c0f2c6acf8b81bf0a0cc29cf02ed6a096c8fbbc4b131f217bb78d",
         detail="648.9M 的 Trainer 未下载（只有 model.pt），因此该点不能精确续训。",
         action="如需从该点续训，按 SHA 从 NUS 补取 trainer.pt；否则用 10004。" ),
    dict(id="nus1080772-milestones-228-288", version=None, cum_decisions=None,
         weights_sha256=None,
         members=[
             dict(version=4433, cum_decisions=228868528,
                  model_sha256="d529f1dbb9aad9fefcb943fab94db0cef195a55412caaeb820c5f33bce05558a",
                  trainer_sha256="550767a509e1f8954d2b4270e58d15991dfcecf59775141a5afdaa8c5cd8dea9"),
             dict(version=4703, cum_decisions=258920789,
                  model_sha256="303d0202e6b91be61030f1ee0b4e02e348990d806e7b5ae0511c720e34ebb8cf",
                  trainer_sha256="21e828a30e6c1243d0ced20414ff5faed740134ce1eeabda170c6b6155f0a78a"),
             dict(version=4972, cum_decisions=288834965,
                  model_sha256="0c6cf183574790771a2679df790ddcc014b492e3347e4bfe6e1d5b11596c1d1a",
                  trainer_sha256="f3a5682f65c4d25023eaca6610ad6761bc3e2a51934f5486feb1d585e7d9236f")],
         detail="1080772 还有三个中间 milestone 有 checkpoint.json（v4433/4703/4972），本地无权重；"
                "318.8M–588.8M 只有诊断/报告，没有可信 checkpoint SHA。",
         action="若要做完整成长曲线，按 model_sha256 从 NUS 原 run 补取；不要补造 318–588M 的 SHA。"),
    dict(id="entity16m-original-4159-255790", version=4159, cum_decisions=None,
         weights_sha256="c8e…（完整 SHA 见 docs/nus/IMPORT_1080772_VERIFIED.md）",
         detail="原版 255790 的 4159 完整状态（模型与 Trainer）不在本地；本地只有 37.0.2 迁移副本。",
         action="可选归档缺口；不得把 110e… 冒充 c8e…。"),
]


def canon_of(path):
    return CANON[str(path)]


def sha_of(path):
    return FILE_SHA[str(path)]


def pick_primary(paths, is_trainer):
    def score(p):
        s = str(p)
        base = os.path.basename(s)
        t = 1 if base == "trainer.pt" else 0
        if "artifacts\\models\\nus-1080772" in s:
            loc = 0
        elif "artifacts\\models\\best" in s:
            loc = 1
        elif "artifacts\\imports" in s:
            loc = 2
        elif "artifacts\\models\\historical" in s:
            loc = 3
        elif "artifacts\\local" in s:
            loc = 4
        else:
            loc = 5
        return (t, loc, len(s))
    return sorted(paths, key=score)[0]


def family_of(params):
    return {
        16436034: "az-entity-16.4m",
        14902579: "az-entity-14.9m",
        14808254: "az-entity-14.8m",
        963635: "az-tiny-0.96m",
        1241354: "legacy-client-1.24m",
        18740981: "autotavern-entity-18.7m",
        2302069: "autotavern-2.3m",
        1048885: "autotavern-1.05m",
        652341: "autotavern-0.65m",
        269175: "autotavern-0.27m",
    }.get(params, f"unknown-{params}")


def main() -> None:
    global CANON, FILE_SHA
    scan = jread(INV / "weight-scan.json")
    canon_rows = jread(INV / "canonical-hashes.json")
    verify = jread(INV / "verify-az.json")
    lineage = jread(INV / "lineage.json")
    info = {r["path"]: r for r in scan["files"]}
    CANON = {r["path"]: r["canon"] for r in canon_rows if r.get("canon")}
    FILE_SHA = {r["path"]: r["file_sha256"] for r in canon_rows}

    verify_by_file = {}
    for m in verify["models"]:
        for p in m["paths"]:
            verify_by_file[p] = m

    groups = {}
    for r in canon_rows:
        if not r.get("canon"):
            continue
        groups.setdefault(r["canon"], []).append(r["path"])

    # trainer records by canonical model hash
    trainers_by_model = {}
    for t in lineage["trainers"]:
        mc = t.get("model_canon")
        if mc:
            trainers_by_model.setdefault(mc, []).append(t)

    roles = {}
    for canon, paths in groups.items():
        is_trainer = [p for p in paths if "optimizer" in ((info.get(p) or {}).get("top_keys") or [])]
        roles[canon] = is_trainer

    entries = []
    for canon, paths in groups.items():
        st = {}
        for p in paths:
            s = (info.get(p) or {}).get("state") or {}
            if s:
                st = s
                break
        params = st.get("n_learnable_like")
        fam = family_of(params)
        c12 = canon[:12]
        spec = SPEC.get(c12, {})
        mid = spec.get("id") or f"{fam}-{c12[:6]}"
        primary = pick_primary(paths, bool(roles[canon]))
        vm = verify_by_file.get(primary)
        if vm is None:
            # try any alias
            for p in paths:
                if p in verify_by_file:
                    vm = verify_by_file[p]
                    break
        load_ok = bool(vm and vm.get("load_ok"))
        fwd = (vm or {}).get("forward") or {}
        fwd_ok = bool(fwd.get("finite")) and fwd.get("shape") == [1, 245]
        card_status = "none"
        card_info = None
        if vm and vm.get("card_static"):
            cs = vm["card_static"]
            card_info = cs
            card_status = "current-3702" if cs.get("exact") else "legacy-pre3702"
        elif vm and vm.get("card_text"):
            ct = vm["card_text"]
            card_info = ct
            card_status = "current-3702" if ct.get("exact") else "legacy-pre3702"

        if fam.startswith("az-"):
            league = ("A-current-3702" if card_status in ("current-3702", "none")
                      else "B-legacy-table")
        elif fam.startswith("legacy-") or fam.startswith("autotavern-"):
            league = "D-archive"
        else:
            league = "D-archive"
        status = spec.get("status") or ("archive-only" if league == "D-archive" else "historical")
        if status in ("local-smoke", "test-fixture"):
            league = "C-engineering"

        entry = {
            "id": mid,
            "name": spec.get("name") or mid,
            "family": fam,
            "status": status,
            "league": league,
            "eval_ready": bool(load_ok and fwd_ok),
            "leaderboard_eligible": bool(load_ok and league in ("A-current-3702", "B-legacy-table")
                                         and status not in ("local-smoke", "test-fixture")),
            "params": params,
            "state_elements": st.get("n_elements"),
            "arch": {
                "kind": st.get("arch"),
                "d_model": st.get("d_model"),
                "n_layers": st.get("n_layers"),
                "dim_ff": st.get("dim_ff"),
                "card_rows": st.get("card_rows"),
                "action_dim": st.get("action_dim"),
                "rank_dim": st.get("rank_dim"),
                "aux_heads": {k[4:]: st[k] for k in
                              ("has_card_head", "has_move_split", "has_strength_head", "has_process_head")
                              if st.get(k)},
            },
            "card_table": {"status": card_status, "diff": card_info},
            "compat": {
                "load_ok": load_ok,
                "load_class": (vm or {}).get("class"),
                "forward_ok": fwd_ok,
                "forward_shape": fwd.get("shape"),
                "error": (vm or {}).get("error"),
                "obs_floats": 3227 if load_ok else None,
                "actions": 245 if load_ok else None,
            },
            "training": {
                "run": spec.get("run"),
                "version": spec.get("version"),
                "cum_decisions": spec.get("cum_decisions"),
            },
            "note": spec.get("note") or "",
            "eval": EVAL.get(c12, []),
            "lineage": None,
            "weights_sha256": canon,
            "file_sha256": sha_of(primary),
            "bytes": (info.get(primary) or {}).get("size"),
            "primary_path": primary,
            "files": [{"path": p, "file_sha256": sha_of(p),
                       "role": ("trainer" if "optimizer" in ((info.get(p) or {}).get("top_keys") or [])
                                else "weights")}
                      for p in paths],
        }
        if c12 in LINEAGE:
            parent, l2, rel, ev = LINEAGE[c12]
            entry["lineage"] = {"parent_weights": parent,
                                "parent_id": None,  # resolved after all entries exist
                                "l2": l2, "relative_l2": rel, "evidence": ev}
        # trainer state attached to this weight payload
        trs = trainers_by_model.get(canon)
        if trs:
            tr = trs[0]
            anchor = (tr.get("anchor_matches_known") or [None])[0]
            entry["trainer"] = {
                "path": tr["path"], "sha256": tr["sha256"],
                "updates": tr.get("updates"),
                "anchor_weights_sha256": tr.get("anchor_canon"),
                "anchor_is_own_model": bool(tr.get("anchor_canon") == canon),
                "anchor_matches_s14_3702": anchor == "fb0911c8cfd97f5bd75344e03b650904ea09f50a81e5f65c4d4f8065db62351d",
                "anchor_nearest_relative_l2": (tr.get("anchor_nearest") or [None, None])[1],
                "has_cuda_rng": bool(((info.get(tr["path"]) or {}).get("trainer") or {}).get("has_cuda_rng")),
                "config_lr": ((info.get(tr["path"]) or {}).get("trainer") or {}).get("config_lr"),
            }
        entries.append(entry)

    # resolve lineage parent / anchor ids now that every entry exists
    by_weights = {e["weights_sha256"]: e for e in entries}
    by_short = {e["weights_sha256"][:12]: e for e in entries}
    for e in entries:
        if e.get("lineage"):
            pw = e["lineage"]["parent_weights"]
            parent = by_weights.get(pw) or by_short.get(pw[:12])
            e["lineage"]["parent_id"] = parent["id"] if parent else None
        tr = e.get("trainer")
        if tr and tr.get("anchor_weights_sha256"):
            a = by_weights.get(tr["anchor_weights_sha256"]) or by_short.get(tr["anchor_weights_sha256"][:12])
            tr["anchor_id"] = a["id"] if a else None

    # stable order: league, family, version
    order = {"production": 0, "approved-resume": 1, "candidate": 2, "reference": 3,
             "historical": 4, "local-smoke": 5, "test-fixture": 6, "archive-only": 7}
    entries.sort(key=lambda e: (order.get(e["status"], 9), e["family"],
                                e["training"].get("version") or 0, e["id"]))

    # ── write library ───────────────────────────────────────────────────
    LIB.mkdir(parents=True, exist_ok=True)
    (LIB / "models").mkdir(exist_ok=True)
    (LIB / "cohorts").mkdir(exist_ok=True)
    sums = []
    for e in entries:
        e["library_model"] = None
        d = None
        src = ROOT / e["primary_path"]
        if e["eval_ready"] and os.path.basename(e["primary_path"]) != "trainer.pt":
            d = LIB / "models" / e["id"]
            d.mkdir(parents=True, exist_ok=True)
            dst = d / "model.pt"
            if not dst.exists():
                os.link(src, dst)
            e["library_model"] = str(dst.relative_to(LIB)).replace("\\", "/")
            sums.append((e["file_sha256"], str(dst.relative_to(LIB)).replace("\\", "/")))
        if e.get("trainer"):
            tsrc = ROOT / e["trainer"]["path"]
            if (tsrc.exists() and e["family"] == "az-entity-16.4m"
                    and e["status"] in ("production", "approved-resume", "candidate")):
                d = LIB / "models" / e["id"]
                d.mkdir(parents=True, exist_ok=True)
                dst = d / "trainer.pt"
                if not dst.exists():
                    os.link(tsrc, dst)
                sums.append((e["trainer"]["sha256"], str(dst.relative_to(LIB)).replace("\\", "/")))
        if d is not None and d.exists():
            (d / "model.json").write_text(
                json.dumps(e, indent=1, ensure_ascii=False), encoding="utf-8")

    cohorts = {
        "mainline-v1": [e["id"] for e in entries if e["league"] == "A-current-3702"
                        and e["status"] in ("production", "approved-resume", "candidate", "reference")],
        "early-v1": [e["id"] for e in entries if e["eval_ready"] and (
            e["family"] in ("az-entity-14.8m", "az-entity-14.9m", "az-tiny-0.96m")
            or (e["family"] == "az-entity-16.4m" and e["league"] == "B-legacy-table"))],
        "forced-smoke-v1": [e["id"] for e in entries if e["status"] == "local-smoke"],
        "archive-v1": [e["id"] for e in entries if e["status"] in ("archive-only", "test-fixture")],
    }
    cohort_doc = {
        "schema": "hsai.model-cohort/v1",
        "note": "同一 cohort 只表示可以放进同一次评测，不代表历史成绩可合并。"
                "历史 protocol_sha256 不同的成绩禁止同榜。",
        "cohorts": {
            "mainline-v1": {
                "members": cohorts["mainline-v1"],
                "historical_protocols": [P_1080772["protocol_sha256"], P_P50["protocol_sha256"]],
                "recommended_protocol": "new unified P50 Dev protocol (to be fixed before run)",
                "warning": "4159/5506/6039 的 4.385/3.781/3.833 与 7357/8255 的 4.469/4.271 属于不同库与协议，不能同榜。",
            },
            "early-v1": {
                "members": cohorts["early-v1"],
                "league": "B-legacy-table / C-engineering mixed",
                "warning": "这些模型内嵌 pre-3702 卡表（minion cost=3 约定）；只能与同为 legacy-table 的模型同榜，"
                           "或单独报告并标注 table=legacy。",
            },
            "forced-smoke-v1": {
                "members": cohorts["forced-smoke-v1"],
                "league": "C-engineering",
                "warning": "本地 1–2 轮工程冒烟，只用于验证管线，不进入棋力榜。",
            },
            "archive-v1": {
                "members": cohorts["archive-v1"],
                "league": "D-archive",
                "warning": "AutoTavern 旧实现或测试夹具；当前 hsbg.az 不能加载，不能直接评测。",
            },
        },
    }
    (LIB / "cohorts" / "cohorts.json").write_text(
        json.dumps(cohort_doc, indent=1, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "schema": "hsai.model-library/v1",
        "generated_at": "2026-11-07",
        "scope": "D:\\HSAI 本地全部 *.pt（不含 .venv）",
        "engine": {
            "game_version": "37.0.2.256589",
            "source_hash": "c7b5f27d6f40d1d2",
            "cards": 3983,
            "obs_floats": 3227,
            "obs_cards": 124,
            "actions": 245,
            "vocab_sha256": "5041e83d810e846b3cbc7d7ff056b84e301472d6ff021ffd1971a18a248e05fb",
            "cards_sha256": "7a02d10794cf80c56fa77b1dd235d49b892c8d3e31c2cf56887f696e2cf922a6",
        },
        "counts": {
            "weight_payloads": len(entries),
            "files_scanned": len(canon_rows),
            "eval_ready": sum(1 for e in entries if e["eval_ready"]),
            "leaderboard_eligible": sum(1 for e in entries if e["leaderboard_eligible"]),
            "with_trainer": sum(1 for e in entries if e.get("trainer")),
            "deduped_files": len(canon_rows) - len(entries),
        },
        "dedup_rule": "同一 weights_sha256（逐张量内容，含卡表 buffer）为一个模型；"
                      "不同容器字节但权重相同的 model.pt/trainer.pt 合并，保留全部 aliases。",
        "compat_rule": {
            "A-current-3702": "能被 hsbg.az 加载、前向 245、内嵌卡表与当前 37.0.2 完全一致",
            "B-legacy-table": "能被 hsbg.az 加载、前向 245，但内嵌 pre-3702 卡表（cost 约定不同）",
            "C-engineering": "可加载，但只是冒烟/测试产物，禁止进棋力榜",
            "D-archive": "旧栈（AutoTavern entities.* / legacy client），当前 hsbg.az 不能加载",
        },
        "models": entries,
        "gaps": GAPS,
    }
    (LIB / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    (LIB / "gaps.json").write_text(
        json.dumps(GAPS, indent=1, ensure_ascii=False), encoding="utf-8")

    with (LIB / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as f:
        for h, p in sorted(sums):
            f.write(f"{h}  {p}\n")

    with (LIB / "index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "family", "status", "league", "params", "version",
                    "cum_decisions", "card_table", "eval_ready", "weights_sha256",
                    "file_sha256", "primary_path"])
        for e in entries:
            w.writerow([e["id"], e["family"], e["status"], e["league"], e["params"],
                        e["training"].get("version"), e["training"].get("cum_decisions"),
                        e["card_table"]["status"], e["eval_ready"], e["weights_sha256"],
                        e["file_sha256"], e["primary_path"]])

    print(json.dumps(manifest["counts"], indent=1))
    print("wrote", LIB)


if __name__ == "__main__":
    main()
