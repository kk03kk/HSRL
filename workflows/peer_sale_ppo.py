"""Terminal-reward PPO on ten sale-logit coefficients of the certified policy."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from workflows.human_turn_preflight import allocation_end, verified_generation
from workflows.peer_finetune_holdout import holdout_protocol, read
from workflows.peer_health_switch import field
from workflows.peer_open_ppo import inspect_trainer
from workflows.peer_ppo_pilot import environment
from workflows.peer_stop_learning import BASE, BASE_SHA

FIT_SEED, EVAL_SEED = 334410000, 337410000
FIT_BATCHES = 24


def create_start(folder):
    import torch
    from hsbg.az.train.peer_actor import load_payload
    payload = torch.load(BASE/"model.pt", map_location="cpu", weights_only=True)
    model, _ = load_payload(payload,"cpu")
    model.enable_sale_adjustment()
    payload["meta"].update(model.checkpoint_metadata())
    payload["state_dict"] = model.state_dict()
    folder.mkdir()
    path = folder/"model.pt"
    torch.save(payload,path)
    restored,_ = load_payload(torch.load(path, map_location="cpu", weights_only=True),"cpu")
    params = {k:p for k,p in restored.named_parameters() if p.requires_grad}
    if (set(params) != {"sale_adjustment.linear.weight","sale_adjustment.linear.bias"}
            or sum(p.numel() for p in params.values()) != 10
            or any(torch.count_nonzero(v).item() for v in params.values())):
        raise ValueError("nonzero initial correction or unexpected trainable set")
    atomic_json(folder/"manifest.json",dict(model=file_identity(path), parent=file_identity(BASE/"model.pt"),
        trainable={k:list(p.shape) for k,p in params.items()}, policy=model.checkpoint_metadata()["sale_adjustment"]))
    atomic_json(folder/"checkpoint.json",dict(model=file_identity(path), kind="zero adjustment weight fork"))
    (folder/"MODEL_CARD.md").write_text("Research derivative of PrentendYan / jimmypretend, CC BY-NC 4.0. "
        "Zero-initialized sale adjustment to the verified D-161 policy.\n",encoding="ascii")
    return path


def run(out):
    import torch
    source = source_identity(ROOT)
    if source["tracked_dirty"] or file_identity(BASE/"model.pt")["sha256"] != BASE_SHA or read(BASE/"verified.json")["games"] != 64:
        raise ValueError("unverified release or parent")
    confirm = []
    for name,n in (("hold",1024),("shift",512)):
        path = Path(f"artifacts/runs/nus-peer-sale-{name}-20261119-1/complete.json")
        row = read(path)
        if row["n"] != n or not row["passed"]:
            raise ValueError("D-161 confirmation does not authorize expansion")
        confirm.append(file_identity(path))
    proto, digest = field()
    hold, _ = holdout_protocol()
    out.mkdir(exist_ok=False)
    initial = create_start(out/"initial")
    stop = allocation_end()-150
    if stop < time.time()+600:
        raise ValueError("insufficient training time")
    settings = environment()
    settings.update(HSRL_ROOT=str(ROOT), HSAI_ARENA_OUT=str((out/"fit").resolve()),
        HSAI_ARENA_CKPT=str(initial.resolve()), HSAI_ACTION_PROFILE="human-turn-3702-standard",
        HSAI_ARENA_SEED=str(FIT_SEED), HSAI_ARENA_ITERATIONS=str(FIT_BATCHES),
        HSAI_ARENA_LR="0.005", HSAI_ARENA_ANCHOR="0.06", HSAI_ARENA_ANCHOR_TAIL="0.04",
        HSAI_ARENA_STOP_EPOCH=str(stop))
    atomic_json(out/"plan.json",dict(source=source, settings=settings, parent=BASE_SHA,
        input=file_identity(initial), confirmations=confirm, development_protocol=proto.sha256(),
        hold_protocol=hold.sha256(), library=digest, evaluation_seed=EVAL_SEED,
        engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"), live_pin=file_identity(ROOT/"config/live/model.json"),
        goal="positive lower 95% utility CI versus frozen retention AND T12 paired log lower bound >= -0.12",
        job=os.environ["SLURM_JOB_ID"]))
    subprocess.run(["bash","deploy/slurm/stage_arena.sbatch"],env=os.environ|settings,check=True)
    final = verified_generation(out/"fit",FIT_SEED)
    if final["version"] != FIT_BATCHES:
        raise ValueError("incomplete pilot")
    model = torch.load(final["model"], map_location="cpu", weights_only=True)
    original = torch.load(initial, map_location="cpu", weights_only=True)["state_dict"]
    changed = {k for k,v in model["state_dict"].items() if not torch.equal(v,original[k])}
    if (model["meta"].get("sale_adjustment") != "low-gold-linear-v1" or
            changed != {"sale_adjustment.linear.weight","sale_adjustment.linear.bias"}):
        raise ValueError("unexpected or missing parameter updates")
    trainer = inspect_trainer(final, initial)
    atomic_json(out/"complete.json",dict(checkpoint=final, trainer=trainer,
        fitted={k:model["state_dict"][k].tolist() for k in sorted(changed)}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    for key in list(os.environ):
        if key.startswith(("HSAI_ARENA_", "HSAI_RECRUIT_START_")):
            del os.environ[key]
    os.environ.update(HSAI_ACTION_PROFILE="human-turn-3702-standard", HSAI_HUMAN_TAIL="last-observed-v1")
    run(args.out.resolve())
