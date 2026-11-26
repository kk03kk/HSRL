"""Write the frozen arena plans for a new 37.0.3 nine-hour run directory."""
import argparse
import json
from pathlib import Path


def prepare(run, assets):
    plans = run / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    for name in ("branches", "dev", "final", "logs"):
        (run / name).mkdir(exist_ok=True)

    def load(path):
        return json.loads((assets / path).read_text())

    def write(name, value):
        with (plans / f"{name}.json").open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")

    hybrid = load("train-hybrid.json")
    script = load("train-script.json")
    human = {"bank": str(assets / "human-train.jsonl"), "mixed": True, "seats": 3,
             "start": 10, "level": 2, "mode": "rolling"}
    write("hybrid", hybrid)
    write("human-late", {**hybrid, "b": {"mixed": False}, "c": human})
    write("stronger", {**hybrid, "c": hybrid["d"]})
    write("script", script)
    write("safe", script)
    development = load("dev-plan.json")
    development.update(load("human-coverage/human-dev-plan.json"))
    write("development", development)
    independent = load("test-plan.json")
    independent.update(load("human-coverage/human-test-plan.json"))
    independent["fresh"] = {**independent["bank"], "bank": str(run / "fresh-bank.jsonl"), "level": 2}
    write("independent", independent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=lambda p: Path(p).resolve())
    parser.add_argument("assets", type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    prepare(args.run, args.assets)
