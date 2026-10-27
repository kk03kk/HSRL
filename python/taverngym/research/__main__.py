import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Paper-inspired bounded experiments, not the authors' released trainer")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--configs", nargs="+", required=True)
    b.add_argument("--device", default="cuda")
    b.add_argument("--smoke", action="store_true")
    b.add_argument("--out", required=True)
    b = sub.add_parser("bank")
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--games", type=int, default=2)
    b.add_argument("--split", choices=("train", "validation", "test"), required=True)
    b.add_argument("--profile", choices=("native", "tempo", "economy", "mixed"), default="native")
    b.add_argument("--out", required=True)
    b = sub.add_parser("bank-report")
    b.add_argument("banks", nargs="+")
    b.add_argument("--out", required=True)
    b = sub.add_parser("exam")
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--games", type=int, default=8)
    b.add_argument("--device", default="cpu")
    b.add_argument("--out", required=True)
    b = sub.add_parser("import-client-bank")
    b.add_argument("input")
    b.add_argument("--split", choices=("train", "validation", "test"), default="test")
    b.add_argument("--max-lines", type=int, default=20000)
    b.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "benchmark":
        from .benchmark import benchmark
        result = benchmark(args.configs, args.out, device=args.device, smoke=args.smoke)
        print(json.dumps({"candidates": len(result["results"]), "report": str(Path(args.out)/'report.json')}))
        return
    if args.command == "bank":
        from .bank import export_natural, summarize
        result = summarize(export_natural(args.out, seed=args.seed, games=args.games, split=args.split, profile=args.profile))
        print(json.dumps(result))
        return
    if args.command == "import-client-bank":
        from .bank import import_client, summarize
        bank = import_client(args.input, args.out, split=args.split, max_lines=args.max_lines)
        print(json.dumps({**summarize(bank), "builds": bank["builds"], "lines_read": bank["lines_read"]}))
        return
    if args.command == "bank-report":
        from .bank import check_splits, summarize
        banks = [json.loads(Path(f).read_text(encoding="utf-8")) for f in args.banks]
        result = {"split_check": check_splits(banks), "summaries": [summarize(b) for b in banks]}
    else:
        from taverngym.offline.evaluate import checkpoint
        from .exam import exam
        cfg, model = checkpoint(args.checkpoint, args.device)
        if args.games < 1:
            p.error("games must be positive")
        result = exam(cfg, model, games=args.games)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out)}))


if __name__ == "__main__":
    main()
