"""Strict experiment configuration and executable/checkpoint identity."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path

import taverngym as tg

SCHEMA = "entity-effects-gru-v1"

# Provisional offline driver budget, not a live-game rule or measured time limit.
TRAIN_ACTION_BUDGET = 200


@dataclass
class Config:
    policy_kind: str = "legacy_gru"
    ff_multiplier: int = 2
    rule_asset_path: str = ""
    rule_asset_sha256: str = ""
    library_bank_path: str = ""
    library_bank_sha256: str = ""
    library_validation_path: str = ""
    library_validation_sha256: str = ""
    bot_profile: str = "native"
    opponent_mode: str = "league"
    decision_budget: int = 0
    anchor_coef: float = 0.0
    anchor_iterations: int = 0
    seed: int = 123
    device: str = "cuda"
    n_games: int = 16
    learner_seats: int = 2
    threads: int = 8
    torch_threads: int = 2
    iterations: int = 1000
    bc_waves: int = 2
    bc_epochs: int = 2
    epochs: int = 3
    seq_len: int = 32
    memory_horizon: int = 128
    chunks_per_batch: int = 8
    gradient_accumulation: int = 1
    dim: int = 128
    layers: int = 3
    heads: int = 4
    hidden: int = 256
    lr: float = 0.0001
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip: float = 0.2
    entropy: float = 0.02
    value_coef: float = 0.5
    rank_coef: float = 0.1
    bc_rank_coef: float = 0.1
    target_kl: float = 0.02
    max_grad_norm: float = 0.5
    history_every: int = 10
    history_size: int = 4
    eval_every: int = 10
    eval_games: int = 32
    eval_seeds: list = field(default_factory=lambda: [900000000, 900100000])
    max_ticks: int = 12000
    rules: dict = field(default_factory=lambda: {
        "content": "Implemented", "record_log": False, "max_turns": 100,
        "turn_action_caps": [TRAIN_ACTION_BUDGET], "max_actions_per_turn": TRAIN_ACTION_BUDGET,
        "hero_bans": [],
        "dark_gift_pair_bans": [['BG36_849', 'BG36_MidGameEffect_000t60'], ['BG23_318', 'BG36_MidGameEffect_000t16'], ['BG36_360t5', 'BG36_MidGameEffect_000t15']],
    })

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        c = cls(**obj)
        c.validate()
        return c

    def validate(self):
        if self.policy_kind not in ("legacy_gru", "compact_gru", "compact_ff", "entity_ff"):
            raise ValueError("unknown policy_kind")
        if self.ff_multiplier not in (2, 4):
            raise ValueError("ff_multiplier must be 2 or 4")
        if self.policy_kind == "entity_ff" and (not self.rule_asset_path or len(self.rule_asset_sha256) != 64):
            raise ValueError("entity_ff requires fingerprinted offline rule assets")
        if self.opponent_mode == "library" and (self.learner_seats != 1 or not self.library_bank_path or len(self.library_bank_sha256) != 64):
            raise ValueError("library requires one learner and a fingerprinted bank")
        if self.bot_profile not in ("native", "tempo", "economy", "mixed") or self.opponent_mode not in ("league", "bots", "library"):
            raise ValueError("unknown opponent profile/mode")
        if (type(self.decision_budget) is not int or self.decision_budget < 0
                or not math.isfinite(self.anchor_coef) or self.anchor_coef < 0
                or type(self.anchor_iterations) is not int or self.anchor_iterations < 0):
            raise ValueError("invalid sample budget or stage anchor")
        for name in ("n_games", "threads", "torch_threads", "seq_len", "chunks_per_batch",
                     "dim", "layers", "heads", "hidden", "iterations", "epochs", "gradient_accumulation",
                     "history_every", "history_size", "eval_every", "eval_games", "max_ticks", "memory_horizon"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.learner_seats <= 8 or self.dim % self.heads:
            raise ValueError("invalid learner_seats or attention dimensions")
        if self.memory_horizon % self.seq_len:
            raise ValueError("memory_horizon must be a multiple of seq_len")
        if not self.eval_seeds or len(set(self.eval_seeds)) != len(self.eval_seeds):
            raise ValueError("eval_seeds must be nonempty and unique")
        if any(not isinstance(s, int) or s < 0 for s in self.eval_seeds):
            raise ValueError("eval seeds must be nonnegative integers")
        if not 0 < self.clip < 1 or self.target_kl <= 0 or self.max_grad_norm <= 0:
            raise ValueError("invalid PPO protection parameters")
        if min(self.entropy, self.value_coef, self.rank_coef, self.bc_rank_coef) < 0:
            raise ValueError("loss coefficients must be nonnegative")
        if self.bc_waves < 0 or self.bc_epochs < 1:
            raise ValueError("invalid BC schedule")
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and self.lr > 0):
            raise ValueError("invalid discount/learning rate")
        if self.rules.get("content", "Implemented") != "Implemented":
            raise ValueError("AllVanilla is prohibited for training")
        if self.rules.get("num_players", 8) != 8 or self.rules.get("experimental_mode", False):
            raise ValueError("profile is eight-seat solo, experimental mode disabled")
        if "NAGA" in self.rules.get("lobby_types", []):
            raise ValueError("Naga is rotated out; cannot force it into a training lobby")
        if self.rules.get("hero_pool"):
            raise ValueError("hero_pool overrides can bypass live profile bans")

    def to_dict(self):
        return asdict(self)


def identity(config):
    """Exclude run duration/device/batch tuning; include all learned semantics."""
    vocab = [tg.card_info(i)[0] for i in range(1, tg.N_CARDS)]
    obj = {"schema": SCHEMA, "engine": tg.game_version(),
           "memory_horizon": config.memory_horizon,
           "engine_source": tg.build_info()["source_hash"], "layout": tg.layout(),
           "vocabulary": vocab, "rules": config.rules,
           "architecture": {k: getattr(config, k) for k in ("dim", "layers", "heads", "hidden")}}
    if config.policy_kind != "legacy_gru":
        obj["schema"] = "compact-entities-v1"
        obj["architecture"].update(policy_kind=config.policy_kind, representation="static66+rule-hash64-v1+owner-pool49")
    if config.policy_kind == "entity_ff":
        obj["schema"] = "entity-ff-lsa-v2"
        obj["architecture"].update(ff_multiplier=config.ff_multiplier,
            representation="static66+tfidf-lsa128+owner-pool49", assets=config.rule_asset_sha256)
    elif config.ff_multiplier != 2:
        obj["architecture"]["ff_multiplier"] = config.ff_multiplier
    if config.opponent_mode == "library":
        obj["opponent_process"] = {"kind":"exogenous-library-v1", "train_bank": config.library_bank_sha256,
            "pool":"external-copies-exempt", "tail":"hold-last", "seat":"one-learner-seven-library"}
    obj["trainer_source"] = hashlib.sha256(b"".join(
        p.name.encode() + p.read_bytes().replace(b"\r", b"")
        for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()
    obj["digest"] = hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
    return obj
