"""Fork the sealed R1 weights into a recoverable public-observation training start."""
import json
from pathlib import Path

from hsbg.runtime.artifacts import atomic_json, file_identity
from workflows.peer_3703_field import R1, R1_SHA

START = Path("artifacts/models/player150-r1-start-20261120-01")


def prepare():
    import hsbg_engine as engine
    import torch
    from hsbg.az.train.common import obs_arrays
    from hsbg.az.train.lobby_filter import make_filtered_game
    from hsbg.az.train.mechanism_obs import observe
    from hsbg.az.train.model import load_checkpoint, save_checkpoint
    from hsbg.az.train.peer_actor import PeerActor
    from hsbg.az.train.player_observation import SCHEMA, SCHEMA_SHA, WIDTH
    from workflows.mirror_hard_gates import _drive_to_turn

    if file_identity(R1)["sha256"] != R1_SHA or not engine.build_is_current("engine"):
        raise ValueError("R1 or native release changed")
    torch.set_num_threads(1)
    torch.manual_seed(541610000)
    old, meta = load_checkpoint(str(R1))
    if not isinstance(old, PeerActor) or meta.get("norm_mean") is not None or meta.get("norm_std") is not None:
        raise ValueError("incompatible parent")
    START.mkdir(exist_ok=False)
    old.enable_player_view()
    model = START / "model.pt"
    save_checkpoint(old, str(model), meta=meta)
    del old
    parent, _ = load_checkpoint(str(R1))
    restored, metadata = load_checkpoint(str(model))
    seen = 0
    with torch.inference_mode():
        for seed in range(541610000, 541610004):
            g, _ = make_filtered_game(engine, seed)
            for turn in (1, 4, 7, 10):
                _drive_to_turn(g, turn)
                if g.is_over() or not g.needs_action(0):
                    continue
                f, c = obs_arrays(observe(g, 0, SCHEMA))
                assert len(f) == WIDTH and json.loads(g.player_card_faces(0))["remaining"] == 150
                features = torch.from_numpy(f)[None]
                cards = torch.from_numpy(c).long()[None]
                mask = torch.tensor(list(g.legal_mask(0)), dtype=torch.bool)[None]
                a = parent(features[:, :3227], cards, mask)
                b = restored(features, cards, mask)
                assert all(torch.equal(x, y) for x, y in zip(a, b))
                seen += 1
    if seen < 8 or metadata["mechanism_schema_sha256"] != SCHEMA_SHA:
        raise ValueError("insufficient or incompatible equivalence evidence")
    record = dict(parent=file_identity(R1), model=file_identity(model), observation=SCHEMA_SHA,
                  equivalence_states=seen, engine=engine.build_info()["source_hash"],
                  optimizer="fresh; later rounds restore full Trainer")
    atomic_json(START / "manifest.json", record)
    atomic_json(START / "checkpoint.json", dict(model=record["model"], trainable_start=True))
    (START / "MODEL_CARD.md").write_text(
        "R1 public-observation fork. Zero-initialized input projections. See manifest.json for verified identities.\n",
        encoding="utf-8")
    print(json.dumps(record), flush=True)
