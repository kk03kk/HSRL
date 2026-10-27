"""本地一键验收：pytest、小规模BC/PPO、回放、成对评测、离线回执；不访问实机。"""
import argparse
import json
from pathlib import Path
import shutil
import time

from local_offline import prepare


def main():
    handles = prepare()
    import pytest
    import torch
    import taverngym as tg
    from taverngym.offline.capture import capture
    from taverngym.offline.trace import read, replay, statistics
    from taverngym.offline.evaluate import checkpoint, compare, selection_protocol, import_metrics
    from taverngym.offline.bridge import MemoryLedger
    from taverngym.offline.sandbox import SimulatorExecutor, decide
    from taverngym.offline.client_audit import audit
    from taverngym.offline.history import PublicHistory
    from taverngym.rl.config import Config, identity
    from taverngym.rl.train import train
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="new directory in this project's runs/")
    p.add_argument("--skip-tests", action="store_true", help="only when pytest was separately run on the same source")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    if not out.is_relative_to((root / "runs").resolve()):
        p.error("validation output must be in main project's runs/")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if not args.skip_tests:
        status = pytest.main([str(root / "tests"), "-q", "--disable-warnings",
                              f"--basetemp={out/'pytest'}", f"--junitxml={out/'tests.xml'}"])
        if status:
            raise SystemExit(status)
    cfg = Config.load(root / "configs/ppo_smoke.json")
    cfg.device = "cpu"
    torch.set_num_threads(1)
    train(cfg, out / "learning")
    cfg, actor = checkpoint(out / "learning/latest.pt")
    cfg.n_games = 1
    result = {"native": tg.build_info(), "identity": identity(cfg), "formal_training": False}
    for name, model in (("bot", None), ("ppo", actor)):
        path = out / f"{name}.jsonl"
        outcome = capture(cfg, seed=73001, actor=model, path=path)
        if outcome["status"] != "completed":
            raise RuntimeError(outcome)
        rows = read(path)
        result[name] = {"outcome": outcome, "replay": replay(rows), "statistics": statistics(rows)}
    _, anchor = checkpoint(out / "learning/bc-anchor.pt")
    comparison = compare(cfg, {"bot": None, "bc": anchor, "ppo": actor, "same_ppo": actor},
                         [1902000000, 1902000001], out=out / "comparison.json")
    for key, value in comparison["paired_rank_differences"].items():
        if key in ("ppo/greedy minus same_ppo/greedy", "ppo/sample minus same_ppo/sample"):
            assert value["mean"] == 0
    selection = selection_protocol(cfg, {"bc": anchor, "ppo": actor}, anchor, games=1)
    (out / "selection-test.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    env = tg.BatchEnv(1, seed=73001, controlled=range(8), config=cfg.rules)
    env.auto_reset = False
    env.step(env.bot_actions())
    executor, ledger = SimulatorExecutor(env, 0, identity(cfg)), MemoryLedger()
    history = PublicHistory()
    history.apply({"type": "game_start", "game_id": "sandbox-0", "revision": 0})
    history_checks = 0
    for _ in range(cfg.max_ticks):
        if not executor.advance_waiting():
            break
        snapshot = executor.observe()
        if history.turn != snapshot.data["turn"]:
            history.apply({"type": "turn_start", "game_id": snapshot.game_id, "revision": snapshot.revision, "turn": snapshot.data["turn"]})
        budget = snapshot.data["policy_context"]["extras"][0][2]
        assert history.values()["actions_this_turn"] == budget[1]
        assert history.values()["freezes_this_turn"] == budget[3]
        history_checks += 1
        if not snapshot.data["ready"]:
            break
        intent = decide(snapshot, actor, cfg, ledger, identity(cfg), allowed_builds=[tg.game_version()], fixture=True)
        receipt = executor.execute(intent)
        ledger.confirm(receipt)
        history.apply({"type": "receipt", "game_id": receipt.game_id, "revision": receipt.after_revision,
                       "intent_id": receipt.intent_id, "turn": snapshot.data["turn"],
                       "status": receipt.status, "kind": intent.semantic["kind"]})
        assert not ledger.confirm(receipt)
    else:
        raise RuntimeError("bridge loop did not terminate naturally")
    assert ledger.count > 0
    final_view = json.loads(env.game(0).view(0))
    result["sandbox"] = {"applied": ledger.count, "real_client_verified": False,
                         "learner_alive": final_view["me"]["alive"], "phase": final_view["phase"],
                         "history_counter_checks": history_checks}
    result["client_audit"] = audit(root / "tests/fixtures/offline/legacy-client.json", ["255130", "37.0.0.255130"])
    result["metrics"] = import_metrics(out / "learning/metrics.jsonl")
    source = root / "python/taverngym/offline/viewer.html"
    shutil.copyfile(source, out / "viewer.html")
    result["seconds"] = time.monotonic() - started
    result["verified"] = True
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(out / 'result.json'), "verified": True, "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
