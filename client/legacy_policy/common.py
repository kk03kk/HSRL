"""在线策略共用的公开信息补全、预判缓存、采用审计与动作核验状态。"""
from collections import Counter
from dataclasses import asdict, replace
from bg_agent.decision_ahead import decision_key
from bg_agent.model import Kind
from legacy_policy.protocol import waiting_for_sync


class CachedLivePolicy:
    participation_fields = ("neural_used", "value_ranked", "value_changed_selection", "actor_changed_selection")

    def _enrich(self, obs):
        catalog = self.legacy.catalog
        def card(c):
            meta = catalog.get(c, obs.build)
            # 只补公开静态卡牌资料；动态日志值优先，绝不接入隐藏实体。
            tags = dict(c.tags)
            for tag in meta.get("mechanics", ()):
                tags.setdefault(tag, "1")
            if str(tags.get("SILENCED", "0")) == "1":
                for tag in ("TAUNT", "DIVINE_SHIELD", "POISONOUS", "VENOMOUS", "REBORN", "WINDFURY",
                            "DEATHRATTLE", "BATTLECRY", "MAGNETIC", "MODULAR", "STEALTH"):
                    tags[tag] = "0"
            if "CARDRACE" not in tags and meta.get("race"):
                tags["CARDRACE"] = meta["race"]
            return replace(c, attack=c.attack if c.attack is not None else meta.get("attack"),
                           health=c.health if c.health is not None else meta.get("health"),
                           tier=c.tier if c.tier is not None else meta.get("techLevel"), tags=tags)
        return replace(obs, board=tuple(card(c) for c in obs.board), hand=tuple(card(c) for c in obs.hand),
                       shop=tuple(card(c) for c in obs.shop), choices=tuple(card(c) for c in obs.choices))

    def _evaluate(self, obs):
        # 缓存键只覆盖策略输入：日志游标/回执/客户端回放推进不应让同一决策重算。
        key = (decision_key(obs), waiting_for_sync(obs))
        with self.lock:
            if key in self.cache:
                decision, trace = self.cache[key]
                self.cache.move_to_end(key)
            else:
                decision, trace = self._compute(obs)
                self.cache[key] = (decision, trace)
                while len(self.cache) > self.config.cache_size:
                    self.cache.popitem(last=False)
            return replace(decision, revision=obs.revision), trace

    def preview(self, obs):
        # 只写有界推理缓存；不写审计、计数或已核验动作状态。
        return self._evaluate(obs)[0]

    def decide(self, obs):
        decision = self._evaluate(obs)[0]
        self.on_decision_adopted(obs, decision)
        return decision

    def on_decision_adopted(self, obs, decision):
        with self.lock:
            _, trace = self._evaluate(obs)
            identity = (obs.game_id, obs.revision, obs.ui_revision, decision.action.key)
            if identity == self.last_adopted:
                return
            self.last_adopted = identity
            self.last_trace = {**trace, "revision": obs.revision, "chosen_key": decision.action.key}
            if decision.action.kind != Kind.WAIT:
                self.counters["decisions"] += 1
                for name in self.participation_fields:
                    self.counters[name] += int(trace.get(name, False))
                self.counters["kind:" + str(decision.action.kind)] += 1
            if self.audit:
                self.audit.write("ai_decision", game_id=obs.game_id, **self.last_trace)

    def on_verified(self, action, before, after, outcome):
        with self.lock:
            self.legacy.on_verified(action, before, after, outcome)
            self.cache.clear()
            if str(outcome.status) in ("confirmed", "resolved"):
                self.counters["verified"] += 1

    def on_abandoned(self, *args):
        with self.lock:
            self.cache.clear()
            self.counters["abandoned"] += 1

    def checkpoint_state(self):
        with self.lock:
            return dict(schema=1, checkpoint_sha256=self.sha256, config=asdict(self.config),
                        counters=dict(self.counters), legacy=self.legacy.checkpoint_state())

    def restore_checkpoint_state(self, state):
        with self.lock:
            if state.get("schema") != 1 or state.get("checkpoint_sha256") != self.sha256 or state.get("config") != asdict(self.config):
                raise ValueError("checkpoint_model_or_config_mismatch")
            self.legacy.restore_checkpoint_state(state["legacy"])
            self.counters = Counter(state["counters"])
            self.cache.clear()
            self.last_adopted = None
            self.last_trace = None
