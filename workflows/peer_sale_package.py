"""Package and trajectory-check the D-161 rule inside a trainable peer checkpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from workflows.peer_finetune_holdout import INITIAL, INITIAL_SHA, read
from workflows.peer_health_switch import field
from workflows.peer_sale_study import RetainMaterial


def package(out):
    import torch
    from hsbg.az.train.peer_actor import load_payload
    from hsbg.az.train.model import load_checkpoint
    source = source_identity(ROOT)
    if source["tracked_dirty"] or file_identity(INITIAL)["sha256"] != INITIAL_SHA:
        raise ValueError("unexpected source or raw wrapper")
    out.mkdir(parents=True, exist_ok=False)
    model = out / "model.pt"
    payload = torch.load(INITIAL, map_location="cpu", weights_only=True)
    payload["meta"]["sale_policy"] = "retain-material-v1"
    probe, _ = load_payload(payload, "cpu")
    torch.save(payload, model)
    restored, _ = load_checkpoint(str(model))
    if not restored.retain_material or any(not torch.equal(v, payload["state_dict"][k])
                                           for k,v in restored.state_dict().items()):
        raise ValueError("policy checkpoint round-trip mismatch")
    if [k for k,p in restored.named_parameters() if p.requires_grad] != [
            k for k,p in probe.named_parameters() if p.requires_grad]:
        raise ValueError("trainable set changed")
    proto, digest = field()
    identity = dict(model=file_identity(model), source=source, initial=file_identity(INITIAL),
        policy="retain-material-v1", engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT/"config/live/model.json"), protocol=proto.sha256(), library=digest,
        reference=read(ROOT/"config/reference-assets.json")["hsbg-ppo"])
    atomic_json(out/"manifest.json", identity)
    atomic_json(out/"checkpoint.json", dict(kind="weight-only policy fork", model=identity["model"]))
    (out/"MODEL_CARD.md").write_text(
        "# Material-retaining peer policy\n\nResearch fork of PrentendYan / jimmypretend's CC BY-NC 4.0 model. "
        "Unchanged D-155 tensors plus observation-based D-161 sale policy. See manifest.json for upstream revisions.\n",
        encoding="ascii")


def check(out, i):
    import torch
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    from hsbg.az.train.eval_policy import make_eval_policy
    from workflows.peer_ppo_panel import trial
    torch.set_num_threads(1)
    proto, _ = field()
    pool = MirrorSnapshotPool(proto.mirror_snaps, sample=proto.mirror_sample)
    def policy(path):
        return make_eval_policy(str(path), protocol=proto, device="cpu")["policy"]
    reference, actor = RetainMaterial(policy(INITIAL)), policy(out/"model.pt")
    class Actions:
        def __init__(self, target):
            self.target = target
        def reset(self):
            self.events = []
            self.target.reset()
        def decide(self, game, seat):
            decision = self.target.decide(game,seat)
            self.events.append((game.turn,seat,decision.action))
            return decision
    a, b = Actions(reference), Actions(actor)
    rows = []
    for j in range(i, 64, 8):
        seed = 321410000+j
        if trial(a, pool, seed, event_turns=True) != trial(b, pool, seed, event_turns=True) or a.events != b.events:
            raise ValueError(f"packaged policy differs on seed {seed}")
        rows.append(dict(seed=seed, decisions=len(a.events), replacements=len(reference.overrides)))
    atomic_json(out/f"identity-{i}.json", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        check(args.out, args.worker)
    else:
        package(args.out)
        def launch(i):
            subprocess.run([sys.executable, __file__, str(args.out), "--worker", str(i)], check=True)
        with ThreadPoolExecutor(8) as executor:
            list(executor.map(launch, range(8)))
        rows = sum((read(args.out/f"identity-{i}.json") for i in range(8)), [])
        if len({r["seed"] for r in rows}) != 64 or not any(r["replacements"] for r in rows):
            raise ValueError("insufficient full-game rule coverage")
        atomic_json(args.out/"verified.json", dict(games=len(rows), actions=sum(r["decisions"] for r in rows),
            affected=sum(r["replacements"]>0 for r in rows), job=os.environ["SLURM_JOB_ID"]))
