"""严格校准库对手：合法结算待选、全部活席注入、逐场核对对位来源。"""
from __future__ import annotations

from collections import Counter
import gzip
import json
import os
from pathlib import Path
import time

from .common import ACTION_SEGMENTS, N_PLAYERS
from .mirror_snapshots import inject_turn

CANCEL = ACTION_SEGMENTS["cancel"][0]
CHOOSE_FIRST, CHOOSE_LAST = ACTION_SEGMENTS["choose"]


class StrictLibrary:
    """同回合同座优先；只从同一冻结校准库补足稀疏座位和极晚回合。"""

    def __init__(self, pool):
        self.pool = pool
        self.by_turn = {
            turn: tuple(seat for seat in range(N_PLAYERS) if pool.has(turn, seat))
            for turn in pool.turns()
        }
        if not self.by_turn:
            raise ValueError("严格校准库为空")
        self.last_turn = max(self.by_turn)
        missing = sorted(set(range(1, self.last_turn + 1)) - self.by_turn.keys())
        if missing:
            raise ValueError(f"严格校准库缺少整个回合：{missing}")

    def sample(self, turn, seat, rng, counts):
        source_turn = min(turn, self.last_turn)
        if source_turn != turn:
            counts["tail_reserve"] += 1
        if self.pool.has(source_turn, seat):
            source_seat = seat
        else:
            source_seat = rng.choice(self.by_turn[source_turn])
            counts["cross_seat"] += 1
        frame = self.pool.get(source_turn, source_seat, rng=rng)
        if frame is not None and source_turn == turn and getattr(frame, 'source_turn', 0) not in (0, turn):
            counts['tail_reserve'] += 1
        return frame


class StrictMirror:
    """每局的已注入席位及真实战斗验收。幽灵保持引擎原有的历史战团规则。"""

    def __init__(self, library, learner, rng, stats, *, frame_rng=None):
        self.library = library
        self.learner = learner
        self.rng = rng
        self.frame_rng = frame_rng
        self.stats = stats
        self.counts = Counter()
        self.turn = 0
        self.audited_turn = 0
        self.learner_active = False
        self.certified = {}
        self.frames = {}

    def get(self, turn, seat, rng):
        # 供既有原子注入与账本检查复用；保存实抽帧用于暴露报告。
        if self.frame_rng is not None:
            rng = self.frame_rng(turn, seat)
        frame = self.library.sample(turn, seat, rng, self.counts)
        self.frames[seat] = frame
        return frame

    def _failure(self, game, exc):
        """只在失败时保存可重放的完整引擎状态，避免只剩一句 worker 异常。"""
        root = Path(os.environ.get("HSRL_FAILURE_DIR", "artifacts/runs/failure-states"))
        path = root / f"strict-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}-{time.time_ns()}.json.gz"
        try:
            from .common import engine_info
            root.mkdir(parents=True, exist_ok=True)
            record = dict(schema="hsrl.strict_failure/v1", error=repr(exc),
                          learner=self.learner, turn=self.turn, counts=dict(self.counts),
                          mirror=self.stats.to_meta(), engine=engine_info(),
                          frames={seat: (dict(path=frame.path, turn=frame.turn,
                                              seed=frame.source_game_seed) if frame is not None else None)
                                  for seat, frame in self.frames.items()},
                          state=json.loads(game.state_json()))
            with gzip.open(path, "xt", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False)
        except Exception as saved:
            return RuntimeError(f"{exc}; 保存失败现场出错: {saved!r}")
        return RuntimeError(f"{exc}; failure_state={path.resolve()}")

    def prepare(self, game):
        try:
            self._prepare(game)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise self._failure(game, exc) from exc

    def _prepare(self, game):
        turn = int(game.turn)
        if turn <= self.turn or game.is_over():
            return
        self.turn = turn
        self.frames.clear()
        health, placements = game.health(), game.placements()
        self.learner_active = health[self.learner] > 0 and placements[self.learner] == 0
        targets = []
        for seat in range(N_PLAYERS):
            if seat == self.learner:
                continue
            if health[seat] <= 0 or placements[seat] != 0:
                self.counts["inactive_skipped"] += 1
                continue
            # 待选时 EndTurn 不合法。只执行一个经掩码验证的引擎选择，
            # 不调用会顺带经营和结束回合的 bot_play。
            for _ in range(64):
                if not game.needs_action(seat):
                    raise RuntimeError(f"严格镜像：活席提前结束 turn={turn} seat={seat}")
                mask = game.legal_mask(seat)
                if mask[0]:
                    break
                action = int(game.bot_action(seat))
                allowed = [a for a in (CANCEL, *range(CHOOSE_FIRST, CHOOSE_LAST + 1))
                           if a < len(mask) and mask[a]]
                if action not in allowed:
                    if not allowed:
                        player = json.loads(game.player_json(seat))
                        raise RuntimeError(f"严格镜像：无法合法结算待选 turn={turn} seat={seat} "
                                           f"bot_action={action} legal={[a for a,v in enumerate(mask) if v]} "
                                           f"gold={player['gold']} pending={player['pending']}")
                    action = allowed[0]
                    self.counts["pending_bot_action_corrected"] += 1
                game.step(seat, action)
                self.counts["pending_choices_resolved"] += 1
                if int(game.turn) != turn or game.is_over():
                    raise RuntimeError(f"严格镜像：待选结算意外推进 turn={turn} seat={seat}")
                health, placements = game.health(), game.placements()
                if health[seat] <= 0 or placements[seat] != 0:
                    self.counts["inactive_skipped"] += 1
                    break
            else:
                raise RuntimeError(f"严格镜像：待选链超限 turn={turn} seat={seat}")
            if health[seat] > 0 and placements[seat] == 0:
                targets.append(seat)
        # 较后结算的待选可能影响前面的席位，注入前重新过滤。
        live = [seat for seat in targets if health[seat] > 0 and placements[seat] == 0]
        self.counts["inactive_skipped"] += len(targets) - len(live)
        applied = []
        before = (self.stats.pool_unbalanced, self.stats.pool_species_unbalanced)
        inject_turn(game, self, turn, seats=live, learner_seat=self.learner,
                    rng=self.rng, stats=self.stats, advance=False, applied_out=applied)
        if (applied != live or before != (self.stats.pool_unbalanced, self.stats.pool_species_unbalanced)
                or self.stats.fallback or self.stats.errors):
            raise RuntimeError(f"严格镜像：活席注入失败 turn={turn} live={live} "
                               f"applied={applied} reasons={self.stats.last_rejected}")
        self.certified.update({seat: turn for seat in live})
        self.counts["eligible_live_seats"] += len(live)

    def after_learner_step(self, game):
        try:
            self._after_learner_step(game)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise self._failure(game, exc) from exc

    def _after_learner_step(self, game):
        turn = self.turn
        if (not self.learner_active or turn <= self.audited_turn
                or (int(game.turn) == turn and not game.is_over())):
            return
        self.audited_turn = turn
        state = json.loads(game.state_json())
        encounters = [combat for combat in state["last_combats"] if combat["turn"] == turn
                      and self.learner in (combat["a"], combat["b"])]
        if len(encounters) > 1 or (not encounters and int(game.turn) > turn
                                   and state["players"][self.learner]["alive"]):
            raise RuntimeError(f"严格镜像：学习席战斗记录不完整 turn={turn}")
        for combat in encounters:
            if combat["b"] is None:
                ghost = state["ghost_player"]
                if ghost is None or ghost not in self.certified:
                    raise RuntimeError(f"严格镜像：幽灵无校准库来源 turn={turn} seat={ghost}")
                self.counts["ghost_combats"] += 1
            else:
                opponent = combat["b"] if combat["a"] == self.learner else combat["a"]
                if self.certified.get(opponent) != turn:
                    raise RuntimeError(f"严格镜像：交手对手未注入 turn={turn} seat={opponent}")
                self.counts["live_combats"] += 1
