"""Small learning verification, not formal training. Run from root after native build."""
import argparse
from pathlib import Path
import json
import torch

from taverngym.rl.config import Config
from taverngym.rl.train import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="new validation directory")
    a = p.parse_args()
    root = Path(a.out)
    if root.exists(): raise FileExistsError("use a new validation directory")
    root.mkdir(parents=True)
    cfg = Config.load("configs/ppo_smoke.json")
    cfg.device = "cpu"
    train(cfg, root / "continuous")
    cfg.iterations = 1
    train(cfg, root / "resumed")
    cfg.iterations = 2
    train(cfg, root / "resumed", resume=True)
    x = torch.load(root / "continuous/latest.pt", weights_only=True)
    y = torch.load(root / "resumed/latest.pt", weights_only=True)
    same_model = all(torch.equal(v, y["model"][k]) for k, v in x["model"].items())
    same_rng = torch.equal(x["rng"]["torch"], y["rng"]["torch"]) and x["rng"]["numpy"] == y["rng"]["numpy"]
    result = {"identical_model": same_model, "identical_rng": same_rng,
              "identical_progress": x["progress"] == y["progress"],
              "continuous_iteration": x["progress"]["iteration"], "resumed_iteration": y["progress"]["iteration"]}
    (root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    assert all(result[k] for k in ("identical_model", "identical_rng", "identical_progress")), result


if __name__ == "__main__": main()
