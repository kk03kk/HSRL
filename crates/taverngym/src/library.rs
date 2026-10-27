//! Exogenous combat-library seats. These are not legal recruit policies.
//! Templates are post end-of-turn, pre start-of-combat; their copies are pool-exempt.
use crate::cards::{CardId, db};
use crate::engine::{Game, Phase};
use crate::rng::Rng;
use crate::state::{Item, Minion, Pending, Player};
use crate::types::EntityId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capture {
    pub turn: u32,
    pub lobby: u16,
    pub deity: Option<CardId>,
    pub paradox: Option<CardId>,
    pub card_table: u64,
    pub players: Vec<Player>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trace {
    pub id: String,
    pub source_group: String,
    #[serde(default = "one")]
    pub weight: u32,
    pub lobby: u16,
    pub deity: Option<CardId>,
    pub paradox: Option<CardId>,
    pub frames: BTreeMap<u32, Player>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bank {
    pub format: String,
    pub split: String,
    pub rules: crate::config::RulesConfig,
    pub card_table: u64,
    pub traces: Vec<Arc<Trace>>,
}

fn one() -> u32 {
    1
}

impl Bank {
    pub fn validate(&self) -> Result<(), String> {
        if self.format != "taverngym.library.v1"
            || self.card_table != db().fingerprint
            || self.traces.is_empty()
        {
            return Err("library format, card table or empty bank".into());
        }
        if !matches!(self.split.as_str(), "train" | "validation" | "test") {
            return Err("library requires explicit split".into());
        }
        for t in &self.traces {
            if t.weight == 0 || t.weight > 1_000_000 {
                return Err("invalid library weight".into());
            }
            if t.frames.keys().next() != Some(&1) {
                return Err("trace must start at turn one".into());
            }
            for (turn, p) in &t.frames {
                if p.hero.is_some_and(|c| c.idx() >= db().len()) {
                    return Err("library hero outside card table".into());
                }

                if *turn == 0
                    || p.hero.is_none()
                    || p.board.len() > 7
                    || p.hand.len() > 10
                    || !p.pending.is_empty()
                {
                    return Err("invalid library frame".into());
                }
                if p.board
                    .iter()
                    .chain(p.hand.iter().filter_map(|h| h.item.as_minion()))
                    .any(|m| {
                        m.card.idx() >= db().len() || m.hp <= 0 || m.atk < 0 || m.max_hp < m.hp
                    })
                {
                    return Err("invalid library minion".into());
                }
            }
            let last = *t.frames.keys().last().unwrap();
            if (1..=last).any(|turn| !t.frames.contains_key(&turn)) {
                return Err("missing intermediate library turn".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Schedule {
    pub learner: usize,
    pub seats: Vec<Option<Arc<Trace>>>,
    pub rounds: u32,
    pub held_tail_frames: u32,
}

impl Game {
    pub fn is_library_seat(&self, p: usize) -> bool {
        self.library
            .as_ref()
            .and_then(|x| x.seats.get(p))
            .is_some_and(Option::is_some)
    }

    pub fn bind_library(&mut self, bank: &Bank, learner: usize) -> Result<(), String> {
        if self.phase != Phase::HeroSelect
            || learner >= self.players.len()
            || self.library.is_some()
            || self.players.iter().any(|p| p.hero.is_some())
        {
            return Err("bind library only at fresh hero selection with one learner".into());
        }
        let mut rules = bank.rules.clone();
        rules.capture_library = false;
        let mut current = self.cfg.clone();
        current.capture_library = false;
        if serde_json::to_value(rules).unwrap() != serde_json::to_value(current).unwrap() {
            return Err("library rules differ from the environment".into());
        }
        let choices: Vec<_> = bank
            .traces
            .iter()
            .filter(|t| {
                t.lobby == self.lobby_races.0 && t.deity == self.deity && t.paradox == self.paradox
            })
            .collect();
        if choices.is_empty() {
            return Err("bank lacks this lobby/deity/paradox context".into());
        }
        // The curriculum stream is independent of combat/shop RNG; no future-state peek.
        let mut rng = Rng::new(self.seed ^ 0x4c49_4252_4152_5931);
        let mut seats = vec![None; self.players.len()];
        for (p, slot) in seats.iter_mut().enumerate() {
            if p == learner {
                continue;
            }
            let total: usize = choices.iter().map(|t| t.weight as usize).sum();
            let mut draw = rng.below(total);
            let selected = choices
                .iter()
                .find(|t| {
                    if draw < t.weight as usize {
                        true
                    } else {
                        draw -= t.weight as usize;
                        false
                    }
                })
                .unwrap();
            let trace = Arc::clone(selected);
            let hero = trace.frames[&1].hero.unwrap();
            self.players[p].pending = vec![Pending::HeroSelect {
                options: vec![hero],
            }];
            *slot = Some(trace);
        }
        self.library = Some(Schedule {
            learner,
            seats,
            rounds: 0,
            held_tail_frames: 0,
        });
        Ok(())
    }

    pub fn apply_library_round(&mut self) {
        let Some(mut schedule) = self.library.take() else {
            return;
        };
        for (p, source) in schedule.seats.iter().enumerate() {
            let Some(trace) = source else {
                continue;
            };
            if !self.players[p].alive {
                continue;
            }
            // Explicit v1 tail policy: hold final observed frame, never invent growth.
            let (&turn, template) = trace.frames.range(..=self.turn).next_back().unwrap();
            schedule.held_tail_frames += u32::from(turn < self.turn);
            let old = self.players[p].clone();
            for m in old
                .board
                .iter()
                .chain(old.hand.iter().filter_map(|h| h.item.as_minion()))
                .chain(old.shop.iter().filter_map(|h| h.item.as_minion()))
                .chain(old.hand_queue.iter().filter_map(Item::as_minion))
            {
                self.pool.return_minion(m);
            }
            let mut next = template.clone();
            let mut ids = HashMap::new();
            for m in &mut next.board {
                remap_minion(self, m, &mut ids);
            }
            for h in &mut next.hand {
                if let Item::Minion(m) = &mut h.item {
                    remap_minion(self, m, &mut ids);
                }
                h.hid = remap_id(self, h.hid, &mut ids);
                h.link = remap_id(self, h.link, &mut ids);
                h.group = remap_id(self, h.group, &mut ids);
                h.src = remap_id(self, h.src, &mut ids);
            }
            for (_, m, _) in &mut next.owed {
                remap_minion(self, m, &mut ids);
            }
            for item in &mut next.hand_queue {
                if let Item::Minion(m) = item {
                    remap_minion(self, m, &mut ids);
                }
            }
            for q in &mut next.hand_queue_marks {
                q.hid = remap_id(self, q.hid, &mut ids);
                q.link = remap_id(self, q.link, &mut ids);
                q.src = remap_id(self, q.src, &mut ids);
            }
            for x in &mut next.scheduled {
                x.target = remap_id(self, x.target, &mut ids);
            }
            for x in &mut next.revert_golden {
                x.0 = remap_id(self, x.0, &mut ids);
            }
            for x in &mut next.hats {
                *x = remap_id(self, *x, &mut ids);
            }
            next.discover_targets.clear();
            next.mishmash_seen = None;
            next.shop.clear();
            next.pending.clear();
            next.ended_turn = true;
            next.id = p;
            next.health = old.health;
            next.armor = old.armor;
            next.alive = old.alive;
            next.placement = old.placement;
            next.last_opponent = old.last_opponent;
            next.recent_opponents = old.recent_opponents;
            next.next_opponent = old.next_opponent;
            next.next_is_ghost = old.next_is_ghost;
            next.stats.combats_won = old.stats.combats_won;
            next.stats.combats_lost = old.stats.combats_lost;
            next.stats.won_last = old.stats.won_last;
            next.stats.lost_last = old.stats.lost_last;
            next.stats.tied_last = old.stats.tied_last;
            next.stats.last_damage = old.stats.last_damage;
            next.stats.streak = old.stats.streak;
            next.stats.last_deity = old.stats.last_deity;
            next.stats.last_returned_max = old.stats.last_returned_max;
            next.scouted = old.scouted;
            next.seen_board = old.seen_board;
            next.sees_all_boards = false;
            next.last_board = next.board.clone();
            next.moves_this_turn = 0;
            next.move_orders.clear();
            self.players[p] = next;
        }
        schedule.rounds += 1;
        self.library = Some(schedule);
    }
}

fn remap_id(g: &mut Game, old: EntityId, ids: &mut HashMap<EntityId, EntityId>) -> EntityId {
    if old == 0 {
        return 0;
    }
    *ids.entry(old).or_insert_with(|| g.alloc_eid())
}

fn remap_minion(g: &mut Game, m: &mut Minion, ids: &mut HashMap<EntityId, EntityId>) {
    m.eid = remap_id(g, m.eid, ids);
    m.pool_copies = 0;
    m.pool_parts.clear();
    m.killed_by = 0;
    m.destroyed = false;
    m.attacked = false;
    m.attacks = 0;
    m.avenge = 0;
    m.mods.clear();
}

/// Cache only exact immutable bank texts; never reuse a bank by a caller-supplied name.
/// Three splits fit the cache; Game snapshots still serialize the selected seat schedules.
pub fn cached_bank(text: &str) -> Result<Arc<Bank>, String> {
    type Cache = Vec<(String, Arc<Bank>)>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .map_err(|_| "library cache poisoned")?;
    if let Some((_, bank)) = cache.iter().find(|(original, _)| original == text) {
        return Ok(Arc::clone(bank));
    }
    let bank: Bank = serde_json::from_str(text).map_err(|e| e.to_string())?;
    bank.validate()?;
    let bank = Arc::new(bank);
    if cache.len() >= 3 {
        cache.remove(0);
    }
    cache.push((text.to_owned(), Arc::clone(&bank)));
    Ok(bank)
}
