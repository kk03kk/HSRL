"""Compare frozen policies from identical T9 element and quilboar openings."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEEDS = range(586710000, 586710012)
CASES = ("elem_budget", "elem_offer", "quill_ready", "quill_cold_search")


def run(args):
    os.environ.update(HSAI_ACTION_PROFILE="human-turn-3703-player150")
    import hsbg_engine as engine
    import torch
    from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
    from hsbg.az.train.lobby_filter import engine_rule_overrides
    from hsbg.runtime.artifacts import atomic_json, file_identity
    from workflows.azerite_course import opening
    from workflows.player150_strategy_eval import ENGINE, R1_SHA, R5_SHA
    from workflows.probe_frozen_late_growth import play

    torch.set_num_threads(1)
    expected = {"r1": R1_SHA, "final": R5_SHA}
    paths = {"r1": args.r1, "final": args.final}
    identities = {name: file_identity(path) for name, path in paths.items()}
    if (engine.build_info()["source_hash"] != ENGINE or not engine.build_is_current(ROOT/"engine")
            or {k: f["sha256"] for k, f in identities.items()} != expected):
        raise ValueError("wrong engine or checkpoint identity")
    protocol = EvalProtocol()
    protocol.assert_training_contract()
    policies = {name:make_eval_policy(str(path), protocol=protocol, device="cpu")["policy"]
                for name,path in paths.items()}
    output = []
    for case in CASES:
        for seed in SEEDS:
            board = json.loads(opening(seed, case).state_json())
            board["cfg"].update(engine_rule_overrides())
            state = json.dumps(board)
            arms = {}
            for name, policy in policies.items():
                game = engine.Game.from_json(state)
                turns = play(seed,case,policy,opening=game,normal_health=True,last_turn=13)
                arms[name] = turns
            output.append(dict(seed=seed, case=case, **arms))
        print(f"completed {case}: {len(SEEDS)} pairs", flush=True)
    atomic_json(args.out, dict(models=identities, engine=ENGINE, protocol=protocol.as_dict(),
                               fixture="T9 controlled opening; health restored to 30 at documented boundaries",
                               rows=output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
