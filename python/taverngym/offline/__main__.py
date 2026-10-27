"""python -m taverngym.offline：本地离线交付入口。"""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--checkpoint")
    c.add_argument("--seed", type=int, default=73001)
    c.add_argument("--games", type=int, default=1)
    c.add_argument("--sample", action="store_true")
    c.add_argument("--out", required=True)
    c = sub.add_parser("replay")
    c.add_argument("trace")
    c = sub.add_parser("report")
    c.add_argument("trace")
    c.add_argument("--out", required=True)
    c = sub.add_parser("viewer")
    c.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--checkpoint", action="append", default=[], help="repeat NAME=PATH; native bot is always included")
    c.add_argument("--seeds", nargs="+", type=int, default=[1902000000, 1902000001])
    c.add_argument("--out", required=True)
    c = sub.add_parser("import-metrics")
    c.add_argument("log")
    c.add_argument("--out", required=True)
    c = sub.add_parser("client-audit")
    c.add_argument("input")
    c.add_argument("--build", action="append", required=True)
    c.add_argument("--out", required=True)
    c = sub.add_parser("selection-eval")
    c.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    c.add_argument("--anchor", required=True)
    c.add_argument("--split", choices=("validation", "test"), default="test")
    c.add_argument("--games", type=int, default=32)
    c.add_argument("--out", required=True)
    c = sub.add_parser("sandbox")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--steps", type=int, default=8)
    c.add_argument("--out", required=True)
    c = sub.add_parser("import-failure")
    c.add_argument("input")
    c.add_argument("--out", required=True)
    c = sub.add_parser("rebuild-history")
    c.add_argument("input")
    c.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "viewer":
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).with_name("viewer.html"), dest)
        print(dest.resolve())
        return
    from . import trace
    if args.command == "rebuild-history":
        from .history import PublicHistory
        history = PublicHistory()
        for line in Path(args.input).read_text(encoding="utf-8").splitlines():
            if line.strip():
                history.apply(json.loads(line))
        result = history.values()
    elif args.command == "import-failure":
        result = trace.import_failure(args.input, args.out)
    elif args.command in ("replay", "report"):
        rows = trace.read(args.trace)
        result = trace.replay(rows) if args.command == "replay" else trace.statistics(rows)
    elif args.command == "sandbox":
        import torch
        import taverngym as tg
        from .evaluate import checkpoint
        from .sandbox import SimulatorExecutor, decide
        from .bridge import MemoryLedger
        from taverngym.rl.config import identity
        cfg, model = checkpoint(args.checkpoint)
        torch.set_num_threads(2)
        env = tg.BatchEnv(1, seed=73001, controlled=range(8), config=cfg.rules)
        env.auto_reset = False
        env.step(env.bot_actions())
        executor, ledger = SimulatorExecutor(env, 0, identity(cfg)), MemoryLedger()
        receipts = []
        if args.steps < 1:
            p.error("sandbox steps must be positive")
        for _ in range(args.steps):
            if not executor.advance_waiting():
                break
            snapshot = executor.observe()
            if not snapshot.data["ready"]:
                break
            intent = decide(snapshot, model, cfg, ledger, identity(cfg), allowed_builds=[tg.game_version()], fixture=True)
            receipt = executor.execute(intent)
            ledger.confirm(receipt)
            receipts.append({"action": intent.action, "semantic": intent.semantic, "status": receipt.status})
        result = {"offline_only": True, "source": "native_fixture", "real_client_verified": False,
                  "applied": ledger.count, "receipts": receipts,
                  "learner_alive": json.loads(env.game(0).view(0))["me"]["alive"]}
    elif args.command == "client-audit":
        from .client_audit import audit
        result = audit(args.input, args.build)
    elif args.command == "selection-eval":
        from .evaluate import checkpoint, selection_protocol
        from taverngym.rl.config import identity
        cfg, anchor = checkpoint(args.anchor)
        policies = {}
        for item in args.checkpoint:
            name, path = item.split("=", 1)
            c, model = checkpoint(path)
            if not name or name in policies or identity(c) != identity(cfg):
                p.error("selection checkpoints require distinct names and matching anchor identity")
            policies[name] = model
        result = selection_protocol(cfg, policies, anchor, games=args.games, split=args.split)
    elif args.command == "import-metrics":
        from .evaluate import import_metrics, metrics_html
        result = import_metrics(args.log)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).with_suffix(".html").write_text(metrics_html(result), encoding="utf-8")
    else:
        import torch
        from taverngym.rl.config import Config, identity
        from .capture import capture
        from .evaluate import checkpoint, compare
        torch.set_num_threads(2)
        if args.command == "capture":
            cfg, actor = checkpoint(args.checkpoint) if args.checkpoint else (Config(device="cpu"), None)
            if args.games < 1:
                p.error("games must be positive")
            cfg.n_games = args.games
            result = capture(cfg, seed=args.seed, actor=actor, greedy=not args.sample, path=args.out)
        else:
            cfg, policies = Config(device="cpu"), {"bot": None}
            expected = None
            for item in args.checkpoint:
                name, path = item.split("=", 1)
                if not name or name in policies:
                    p.error("policy names must be distinct")
                candidate_cfg, actor = checkpoint(path)
                current = identity(candidate_cfg)["digest"]
                if expected is not None and current != expected:
                    p.error("checkpoints from different identities require separate comparisons")
                cfg, expected, policies[name] = candidate_cfg, current, actor
            result = compare(cfg, policies, args.seeds, out=args.out)
    if hasattr(args, "out") and args.command not in ("capture", "compare", "import-failure"):
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result.get("status") in ("failed", "truncated"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
