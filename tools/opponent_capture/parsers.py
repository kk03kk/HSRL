"""Independent parsers of public Power/HDT log formats, standard library only.

No old-project imports. Unknown fields stay unknown. Serialized parser state never
contains player names, account identifiers or raw log lines.
"""
import copy
import hashlib
import json
import re


VERSION = "encounter-parser-v1"
KEYWORDS = {"Taunt": "TAUNT", "Div": "DIVINE_SHIELD", "Divine Shield": "DIVINE_SHIELD",
            "Poison": "POISONOUS", "Poisonous": "POISONOUS", "Ven": "VENOMOUS",
            "Venomous": "VENOMOUS", "Reborn": "REBORN", "Windfury": "WINDFURY",
            "Mega-Windfury": "MEGA_WINDFURY", "Stealth": "STEALTH", "Cleave": "CLEAVE"}
NUMERIC_TAG = re.compile(r"^[A-Z][A-Z0-9_]*$|^\d+$")
PRIVATE_TAG = re.compile(r"ACCOUNT|BATTLETAG|BATTLE_TAG|PLAYER_NAME|REAL_NAME", re.I)


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def blank_side():
    return {"board": [], "hero_id": None, "hero_power_id": None, "tier": None,
            "health": None, "armor": None, "hand": None, "hand_scope": "unknown"}


class Catalog:
    def __init__(self, data):
        self.version = data.get("meta", {}).get("client_version")
        self.cards = {c["id"]: c for c in data["cards"]}
        self.names = {}
        for c in data["cards"]:
            if c.get("kind") != "minion" or c.get("normal"):
                continue
            for name in (c.get("name"), c.get("name_zh")):
                if name:
                    self.names.setdefault(name.casefold(), set()).add(c["id"])

    def identify(self, name, golden):
        ids = self.names.get(name.casefold(), set())
        if len(ids) != 1:
            return None
        c = self.cards[next(iter(ids))]
        return c.get("golden") or None if golden else c["id"]


def record_base(source, root, index, kind):
    return {"format": "taverngym.encounter.v1", "parser": VERSION,
            "encounter_id": fingerprint([root, index, kind]), "root": root,
            "source_id": source, "source_kind": kind, "combat_index": index,
            "build": None, "rules_patch": None, "turn": None, "turn_raw": None,
            "available_tribes": None, "matchmaking_strength": "unmeasured",
            "player": blank_side(), "opponent": blank_side(), "events": [],
            "actual_result": None, "second_implementation": None,
            "capture_complete": False, "policy_input": False,
            "training_import_allowed": False, "human_calibrated": False,
            "own_state_role": "diagnostic_context_only", "missing_fields": []}


def assess(row):
    """Usable observed boards and native training compatibility are different gates."""
    declared = set(row.get("missing_fields", [])) - set(row.get("derived_missing_fields", []))
    missing = list(declared)
    for side in ("player", "opponent"):
        band = row[side]
        if not band.get("board"):
            missing.append(side + ".board_empty_or_unobserved")
        for i, c in enumerate(band.get("board", [])):
            if not c.get("card_id"):
                missing.append(f"{side}.board.{i}.card_id")
            if any(c.get(k) is None for k in ("attack", "health")):
                missing.append(f"{side}.board.{i}.stats")
            if c.get("unparsed_effect_hashes"):
                missing.append(f"{side}.board.{i}.unparsed_effects")
        slots = [c.get("slot") for c in band.get("board", [])]
        if len(slots) > 7 or len(slots) != len(set(slots)):
            missing.append(side + ".inconsistent_board_slots")
        if band.get("hero_id") is None:
            missing.append(side + ".hero_id")
        for key in ("hero_power_id", "tier", "health", "armor"):
            if band.get(key) is None:
                missing.append(side + "." + key)
        if band.get("hand_scope") != "complete":
            missing.append(side + ".unrevealed_hand_or_counters")
    if row.get("turn") is None:
        missing.append("turn")
    if row.get("build") is None:
        missing.append("client_build")
    if row.get("available_tribes") is None:
        missing.append("available_tribes")
    if row.get("actual_result") is None:
        missing.append("actual_result")
    row["missing_fields"] = sorted(set(missing))
    row["derived_missing_fields"] = sorted(set(missing) - declared)
    row["combat_signature"] = fingerprint([row["root"], row.get("turn"), row["player"]["board"], row["opponent"]["board"]])
    opp = row["opponent"]["board"]
    row["opponent_board_recorded"] = bool(opp) and all(
        c.get("card_id") and c.get("attack") is not None and c.get("health") is not None for c in opp)
    row["readiness"] = "observed_board_needs_adapter" if row["opponent_board_recorded"] else "incomplete_observation"
    row["solo_training_candidate"] = bool(row["opponent_board_recorded"] and row.get("mode") == "GT_BATTLEGROUNDS"
        and row.get("turn") and row.get("build") and not any("inconsistent" in m or "ambiguous" in m for m in missing))
    return row


class PowerParser:
    def __init__(self, source, state=None):
        self.source = source
        self.s = state or {"entities": {}, "players": {}, "seq": 0, "combat": 0,
                           "game": False, "bg": False, "mode": None, "build": None,
                           "local": None, "pending": None, "active": None,
                           "phase": False, "seen_create": False, "lines": 0,
                           "alias": {}, "non_bg_games": 0, "gaps": 0}

    def entity(self, eid):
        return self.s["entities"].setdefault(str(eid), {"card_id": "", "tags": {}})

    def ref(self, value):
        value = value.strip()
        if value.startswith("Entity="):
            value = value[len("Entity="):]
        m = re.search(r"\bid=(\d+)", value)
        if m:
            eid = int(m[1]); ent = self.entity(eid)
            card = re.search(r"\bcardId=([^\s\]]*)", value)
            if card and card[1] and not ent["card_id"]:
                ent["card_id"] = card[1]
            for field, tag in (("player", "CONTROLLER"), ("zonePos", "ZONE_POSITION"), ("zone", "ZONE")):
                f = re.search(r"\b" + field + r"=([^\s\]]+)", value)
                if f:
                    # Reference strings describe a historical entity, not a fresh state delta.
                    # Replacing tags here resurrects dead/shop entities on later references.
                    ent["tags"].setdefault(tag, f[1])
            return eid
        if value.isdigit():
            return int(value)
        if value == "GameEntity":
            return self.s.get("game_entity", 1)
        return self.s["alias"].get(fingerprint(value))

    def start(self):
        if self.s["active"]:
            return
        self.s["combat"] += 1
        self.s["active"] = {"index": self.s["combat"], "row": None, "events": [],
                             "marker_seen": self.s["phase"], "start_line": self.s["lines"]}

    def side(self, controller):
        band = blank_side()
        if controller is None:
            return band
        for eid, ent in self.s["entities"].items():
            t = ent["tags"]
            if integer(t.get("CONTROLLER")) != controller:
                continue
            if t.get("CARDTYPE") in ("HERO", "3") and t.get("ZONE") in ("PLAY", "1"):
                band.update(hero_id=ent["card_id"] or None, health=integer(t.get("HEALTH")),
                            damage=integer(t.get("DAMAGE", "0")), armor=integer(t.get("ARMOR")),
                            tier=integer(t.get("PLAYER_TECH_LEVEL")))
            if t.get("CARDTYPE") in ("HERO_POWER", "10") and t.get("ZONE") in ("PLAY", "1"):
                band["hero_power_id"] = ent["card_id"] or None
            if (t.get("CARDTYPE") in ("MINION", "4") and t.get("ZONE") in ("PLAY", "1")
                    and (integer(t.get("ZONE_POSITION")) or 0) > 0):
                hp, damage = integer(t.get("HEALTH")), integer(t.get("DAMAGE", "0"))
                band["board"].append({"entity_id": int(eid), "card_id": ent["card_id"] or None if not ent.get("hidden") else None,
                    "slot": integer(t.get("ZONE_POSITION")), "attack": integer(t.get("ATK")),
                    "health": hp - damage if hp is not None and damage is not None else None,
                    "max_health": hp, "golden": t.get("PREMIUM") == "1",
                    "keywords": [k for k in KEYWORDS.values() if t.get(k) == "1"],
                    "observed_tags": dict(t)})
        band["board"].sort(key=lambda c: (c["slot"], c["entity_id"]))
        # Preserve observed attachments/counters rather than reducing a warband to stats.
        for c in band["board"]:
            c["enchantments"] = [{"card_id": e["card_id"] or None, "observed_tags": dict(e["tags"])}
                for e in self.s["entities"].values() if not e.get("hidden") and
                integer(e["tags"].get("ATTACHED")) == c["entity_id"]]
        band["hand"] = [{"card_id": e["card_id"], "attack": integer(e["tags"].get("ATK")),
                          "health": integer(e["tags"].get("HEALTH")), "observed_tags": dict(e["tags"])}
            for e in self.s["entities"].values() if integer(e["tags"].get("CONTROLLER")) == controller
            and e["tags"].get("ZONE") in ("HAND", "3") and e["tags"].get("CARDTYPE") in ("MINION", "4")
            and e["card_id"] and not e.get("hidden")]
        band["hand_scope"] = "revealed_minions_only_not_full_hand"
        band["public_combat_objects"] = [{"card_id": e["card_id"], "observed_tags": dict(e["tags"])}
            for e in self.s["entities"].values() if integer(e["tags"].get("CONTROLLER")) == controller
            and e["tags"].get("ZONE") == "SECRET" and e["card_id"] and not e.get("hidden")]
        player = self.s["entities"].get(str(self.s["players"].get(str(controller))), {}).get("tags", {})
        band["observed_player_tags"] = dict(player)
        if band["tier"] is None:
            band["tier"] = integer(player.get("PLAYER_TECH_LEVEL"))
        if band["health"] is not None:
            band["health"] -= band.get("damage") or 0
        return band

    def snapshot(self):
        active = self.s["active"]
        root = fingerprint([self.source, self.s["seq"]])
        row = record_base(self.source, root, active["index"], "power_log")
        local = self.s["local"]
        local_source = "explicit_player_metadata"
        if local is None:
            # Each recruit controller has visible market controls; the opponent does not.
            # DebugPrintGame lists BOTH PlayerIDs, so its first/last row is not local identity.
            controls = set()
            for ent in self.s["entities"].values():
                card, t = ent["card_id"], ent["tags"]
                if card in {"TB_BaconShop_8p_Reroll_Button", "TB_BaconShopLockAll_Button", "TB_BaconShop_DragSell"}:
                    c = integer(t.get("CONTROLLER"))
                    if c is not None:
                        controls.add(c)
            if len(controls) == 1:
                local = next(iter(controls)); local_source = "unique_shop_control_owner"
        controllers = set()
        for ent in self.s["entities"].values():
            t = ent["tags"]
            if t.get("ZONE") in ("PLAY", "1") and t.get("CARDTYPE") in ("MINION", "4", "HERO", "3"):
                c = integer(t.get("CONTROLLER"))
                if c is not None and c != local:
                    controllers.add(c)
        opponent = next(iter(controllers)) if len(controllers) == 1 else None
        raw_turn = integer(self.entity(self.s.get("game_entity", 1))["tags"].get("TURN"))
        row.update(build=self.s["build"], turn_raw=raw_turn,
                   turn=(raw_turn + 1) // 2 if raw_turn and raw_turn > 0 else None,
                   turn_source="game_half_turn", player=self.side(local), opponent=self.side(opponent),
                   boundary="post_start_before_first_attack", local_controller=local,
                   local_controller_source=local_source if local is not None else None,
                   opponent_controller=opponent, mode=self.s["mode"],
                   source_line=active["start_line"], phase_marker_seen=active["marker_seen"])
        row["observed_game_tags"] = dict(self.entity(self.s.get("game_entity", 1))["tags"])
        row["opponent_public_seat"] = integer(row["player"].get("observed_player_tags", {}).get("NEXT_OPPONENT_PLAYER_ID"))
        if local is None:
            row["missing_fields"].append("local_player_id")
        if opponent is None:
            row["missing_fields"].append("opponent_controller_ambiguous")
        if not self.s["seen_create"] or self.s["gaps"]:
            row["missing_fields"].append("incomplete_log_history")
        if any(len(row[x]["board"]) > 7 for x in ("player", "opponent")):
            row["missing_fields"].append("board_capacity_or_playback_inconsistent")
        return assess(row)

    def finish(self, reason):
        active = self.s["active"]
        if active is None:
            return []
        row = active["row"] or self.snapshot()
        row["capture_complete"] = reason == "phase_end"
        row["boundary_end"] = reason
        row["events"] = active["events"]
        if active["row"] is None:
            row["boundary"] = "no_attack_boundary_snapshot"
            row["missing_fields"].append("pre_attack_snapshot")
        self.s["active"] = None
        return [assess(row)]

    def feed(self, raw):
        self.s["lines"] += 1
        emitted = []
        # GameState and PowerTaskList duplicate events; only the server-state channel is used.
        if "GameState.DebugPrintGame() - " in raw:
            msg = raw.split("GameState.DebugPrintGame() - ", 1)[1].strip()
            for field, key in (("BuildNumber", "build"), ("PlayerID", "local")):
                m = re.fullmatch(field + r"=(\d+)", msg)
                if m:
                    self.s[key] = int(m[1]) if key == "local" else m[1]
            alias = re.match(r"PlayerID=(\d+), PlayerName=(.*)", msg)
            if alias and alias[1] in self.s["players"]:
                self.s["alias"][fingerprint(alias[2].strip())] = self.s["players"][alias[1]]
            m = re.fullmatch(r"GameType=(\w+)", msg)
            if m:
                self.s["mode"] = m[1]
                self.s["bg"] = m[1] == "GT_BATTLEGROUNDS"
                if not self.s["bg"]:
                    self.s["non_bg_games"] += 1
            return emitted
        if "GameState.DebugPrintPower() - " not in raw:
            return emitted
        msg = raw.split("GameState.DebugPrintPower() - ", 1)[1].strip()
        if msg == "CREATE_GAME":
            emitted.extend(self.finish("new_game"))
            for key, value in (("entities", {}), ("players", {}), ("alias", {}), ("combat", 0),
                               ("phase", False), ("pending", None), ("local", None),
                               ("mode", None), ("bg", False), ("seen_create", True), ("gaps", 0)):
                self.s[key] = value
            self.s["seq"] += 1
            return emitted
        m = re.match(r"GameEntity EntityID=(\d+)", msg)
        if m:
            self.s["pending"] = int(m[1]); self.s["game_entity"] = int(m[1]); self.entity(int(m[1]))
            return emitted
        m = re.match(r"Player EntityID=(\d+) PlayerID=(\d+)", msg)
        if m:
            self.s["pending"] = int(m[1]); self.s["players"][m[2]] = int(m[1]); self.entity(int(m[1]))
            return emitted
        m = re.match(r"(?:FULL|SHOW|CHANGE)_ENTITY - (?:Creating ID=(\d+)|Updating (.+?)) CardID=(\S*)", msg)
        if m:
            eid = int(m[1]) if m[1] else self.ref(m[2])
            if eid is not None:
                self.s["pending"] = eid; self.entity(eid)["card_id"] = m[3]
                self.entity(eid)["hidden"] = False
            return emitted
        hide = re.match(r"HIDE_ENTITY - Entity=(.+?) tag=(\w+) value=(\S+)", msg)
        m = hide or re.match(r"TAG_CHANGE Entity=(.+?) tag=(\w+) value=(\S+)", msg)
        if m:
            self.s["pending"] = None
            eid, tag, value = self.ref(m[1]), m[2], m[3]
            if hide and eid is not None:
                self.entity(eid)["hidden"] = True
        else:
            m = re.match(r"tag=(\w+) value=(\S+)", msg)
            if not m or self.s["pending"] is None:
                eid = None
            else:
                eid, tag, value = self.s["pending"], m[1], m[2]
        if eid is not None:
            # Only numeric values and small known enum tokens; never persist arbitrary strings.
            if not PRIVATE_TAG.search(tag) and (integer(value) is not None or value in {"GAME", "PLAYER", "HERO", "MINION", "HERO_POWER",
                                                       "SPELL", "ENCHANTMENT", "PLAY", "HAND", "DECK", "SETASIDE",
                                                       "GRAVEYARD", "REMOVEDFROMGAME", "SECRET", "INVALID"}):
                self.entity(eid)["tags"][tag] = value
            if tag == "BACON_IN_COMBAT_PHASE" and self.s["bg"]:
                if value == "1":
                    self.s["phase"] = True; self.start()
                elif value == "0":
                    emitted.extend(self.finish("phase_end")); self.s["phase"] = False
            if self.s["active"] and tag in ("ATK", "HEALTH", "DAMAGE", "ZONE", "DIVINE_SHIELD", "REBORN"):
                a = self.s["active"]
                if len(a["events"]) < 4000:
                    a["events"].append({"line": self.s["lines"], "type": "tag", "entity": eid, "tag": tag, "value": value})
                else:
                    if a["row"]:
                        a["row"]["missing_fields"].append("event_limit")
            return emitted
        if "BLOCK_START BlockType=ATTACK" in msg and self.s["bg"]:
            self.s["pending"] = None
            if not self.s["active"]:
                self.start()
            a = self.s["active"]
            if a["row"] is None:
                a["row"] = self.snapshot()
                a["row"]["boundary_end"] = "waiting"
                emitted.append(copy.deepcopy(a["row"]))
            a["events"].append({"line": self.s["lines"], "type": "attack_block"})
        if msg.startswith(("BLOCK_", "META_DATA", "HIDE_ENTITY")):
            self.s["pending"] = None
        return emitted

    def pending_record(self):
        a = self.s["active"]
        if a and a["row"]:
            row = copy.deepcopy(a["row"]); row["events"] = a["events"]
            return row
        return None


class HDTParser:
    def __init__(self, source, catalog, state=None):
        self.source, self.catalog = source, catalog
        self.s = state or {"seq": 0, "game": 0, "turn": None, "active": None,
                           "input": False, "side": None, "lines": 0, "gaps": 0}

    def card(self, body):
        m = re.fullmatch(r"(.+?)\s+(-?\d+)/(-?\d+)(.*)", body.strip())
        if not m:
            return None
        name, attack, hp, rest = m.groups()
        parts = [x.strip() for x in re.split(r",(?![^\[]*\])", rest.lstrip(", ")) if x.strip()]
        golden = any(x in ("G", "Golden") for x in parts)
        card_id = self.catalog.identify(name, golden)
        tags, unknown = {}, []
        for part in parts:
            if part in KEYWORDS or part in ("G", "Golden"):
                continue
            if "=" in part:
                key, value = part.split("=", 1)
                if NUMERIC_TAG.fullmatch(key.strip()) and not PRIVATE_TAG.search(key) and integer(value.strip()) is not None:
                    tags[key.strip()] = integer(value.strip()); continue
            # Do not save free-form log text; record a diagnostic hash instead.
            unknown.append(fingerprint(part)[:16])
        return {"card_id": card_id, "catalog_name": self.catalog.cards[card_id].get("name") if card_id else None,
                "unresolved_name_hash": fingerprint(name) if card_id is None else None,
                "attack": int(attack), "health": int(hp),
                "golden": golden, "keywords": [KEYWORDS[p] for p in parts if p in KEYWORDS],
                "observed_tags": tags, "unparsed_effect_hashes": unknown,
                "id_source": "unique_reference_name" if card_id else "unresolved"}

    def finish(self, reason):
        row = self.s["active"]
        if row is None:
            return []
        if not row.get("input_closed"):
            row["missing_fields"].append("incomplete_simulation_input")
        row["capture_complete"] = bool(row.get("input_closed"))
        row["boundary_end"] = reason
        self.s["active"] = None; self.s["input"] = False; self.s["side"] = None
        return [assess(row)]

    def feed(self, raw):
        self.s["lines"] += 1
        msg = raw.split(" >> ", 1)[-1].strip()
        out = []
        if "CardDefsManager.EnsureLatestCardDefs" in raw:
            build = re.search(r"\bBuild[=: ]+(\d{6})\b", msg)
            if build:
                self.s["reference_data_build"] = build[1]
        if "HandleGameStart" in raw or "Game start" == msg:
            out.extend(self.finish("game_boundary")); self.s["game"] += 1; self.s["turn"] = None
        turn = re.fullmatch(r"--- (?:Player|Opponent) turn (\d+) ---", msg)
        if turn:
            value = int(turn[1])
            if self.s["turn"] is not None and value < self.s["turn"]:
                out.extend(self.finish("turn_reset")); self.s["game"] += 1
            self.s["turn"] = value
        if "----- Simulation Input -----" in msg:
            out.extend(self.finish("next_simulation")); self.s["seq"] += 1
            # Whole source session shares a split root; do not leak repeats of one game.
            row = record_base(self.source, self.source, self.s["seq"], "hdt_bobsbuddy")
            row.update(turn=self.s["turn"], source_line=self.s["lines"],
                       boundary="bobsbuddy_input_phase_not_proven", mode="battlegrounds_unverified_solo",
                       reference_data_build=self.s.get("reference_data_build"), reference_catalog=self.catalog.version,
                       combat_index_source="simulation_input_sequence_not_unique_battle_id")
            row["missing_fields"] += ["input_phase_not_proven", "server_rules_patch", "hidden_counters"]
            self.s["active"] = row; self.s["input"] = True; self.s["side"] = None
            return out
        row = self.s["active"]
        if row is None:
            return out
        if "----- End of Input -----" in msg:
            row["input_closed"] = True; self.s["input"] = False
            out.append(assess(copy.deepcopy(row)))
        if self.s["input"]:
            team = re.match(r"^(PlayerTeammate|OpponentTeammate):\s*(.*)", msg)
            if team:
                self.s["side"] = None
                if team[2] != "null":
                    row["missing_fields"].append("unsupported_team_context")
                    row["mode"] = "team_context_not_solo"
                return out
            side = re.match(r"^(Player|Opponent):\s*heroPower=([A-Za-z0-9_]+)(.*)", msg)
            if side:
                self.s["side"] = side[1].lower(); band = row[self.s["side"]]
                band["hero_power_id"] = side[2]
                used = re.search(r"used=(True|False)", side[3])
                data = re.search(r"data=(-?\d+)", side[3])
                band["hero_power_used"] = used[1] == "True" if used else None
                band["hero_power_data"] = int(data[1]) if data else None
                return out
            if msg.startswith("Hand:") and self.s["side"]:
                band = row[self.s["side"]]; cards = re.findall(r"\[([^\[\]]*)\]", msg[5:])
                band["hand"] = [c for body in cards if (c := self.card(body))]
                band["hand_scope"] = "simulator_visible_subset"
                if msg[5:].strip() and not cards:
                    row["missing_fields"].append(self.s["side"] + ".unparsed_hand")
                return out
            if msg.startswith("[") and msg.endswith("]") and self.s["side"]:
                band = row[self.s["side"]]; c = self.card(msg[1:-1])
                if c:
                    c["slot"] = len(band["board"]) + 1; band["board"].append(c)
                else:
                    row["missing_fields"].append(self.s["side"] + ".unparsed_board_entry")
        result = re.search(r"ValidateSimulationResult\w*.*?\bresult=(Win|Loss|Tie)\b", raw)
        if result:
            row["actual_result"] = {"outcome": result[1].lower(), "source": "hdt_reported_client_result",
                                    "linkage": "between_simulation_inputs_not_server_event_id"}
            out.append(assess(copy.deepcopy(row)))
        rates = re.search(r"WinRate=([\d.]+)%.*?TieRate=([\d.]+)%.*?LossRate=([\d.]+)%", msg)
        if rates:
            row["second_implementation"] = {"name": "BobsBuddy", "official": False,
                "win": float(rates[1]) / 100, "tie": float(rates[2]) / 100, "loss": float(rates[3]) / 100}
            out.append(assess(copy.deepcopy(row)))
        return out

    def pending_record(self):
        return assess(copy.deepcopy(self.s["active"])) if self.s["active"] and self.s["active"].get("input_closed") else None
