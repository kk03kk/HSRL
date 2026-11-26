use serde::Serialize;
use taverngym::cards::{CardId, db};
use taverngym::content::is_bounty;
use taverngym::state::{Continuation, Item, Minion, Pending, Player};
use taverngym::types::Kw;
use taverngym::{Action, Game, cardid};

use super::catalog::{Route::*, *};

#[derive(Default)]
pub struct Coverage {
    pub selected: Option<u32>,
    pub core: Option<u32>,
    pub started: Option<u32>,
    pub full: Option<u32>,
    pub started_full: Option<u32>,
    pub final_full: bool,
    pub gain: i32,
    high_spend: bool,
    pub best: Option<Snapshot>,
}

#[derive(Default)]
pub struct Log {
    pub routes: [Coverage; 25],
    entered: Option<Route>,
    last_combat: u32,
    combat_witness: Option<Witness>,
}

#[derive(Serialize)]
pub struct Snapshot {
    pub turn: u32,
    pub tavern: u8,
    pub hero: Option<&'static str>,
    pub full: bool,
    pub started: bool,
    pub attack: i64,
    pub health: i64,
    pub board: Vec<MinionRow>,
}

#[derive(Serialize)]
pub struct MinionRow {
    pub name: &'static str,
    pub card: &'static str,
    pub golden: bool,
    pub attack: i32,
    pub health: i32,
}

pub struct Witness {
    route: Route,
    metric: [i64; 3],
    action: Action,
    spell: Option<CardId>,
    anchor: Option<(u32, i32, i64)>,
    others: [(u32, i64); 7],
}

fn strength(m: &Minion) -> i64 {
    i64::from(m.atk) + i64::from(m.hp)
}
fn anchor(route: Route, p: &Player) -> Option<(u32, i32, i64)> {
    p.board
        .iter()
        .find(|m| id(m) == route.spec().anchor)
        .map(|m| (m.eid, m.counter, strength(m)))
}
fn stat(route: Route, p: &Player) -> [i64; 3] {
    let s = &p.stats;
    let shop = || {
        p.shop
            .iter()
            .filter_map(|s| s.item.as_minion())
            .map(|m| i64::from(m.gems))
            .sum()
    };
    let hand = || {
        p.hand
            .iter()
            .filter_map(|h| h.item.as_minion())
            .filter(|m| m.is(route.race()))
            .map(strength)
            .sum()
    };
    let deity = || {
        p.deity
            .as_ref()
            .map_or(0, |d| i64::from(d.atk) + i64::from(d.hp))
    };
    let magnets = || {
        p.board
            .iter()
            .map(|m| {
                m.attached
                    .iter()
                    .filter(|&&c| taverngym::content::VOLUMIZERS.contains(&key(c)))
                    .count() as i64
            })
            .sum()
    };
    match route {
        Lobster => [i64::from(s.lobster_bonus), 0, 0],
        UndeadAttack => [i64::from(s.undead_atk), 0, 0],
        DemonShop | DemonDeath => [
            i64::from(s.tavern_atk) + i64::from(s.tavern_hp),
            p.shop.len() as i64,
            0,
        ],
        Beetle => [i64::from(s.beetle_atk) + i64::from(s.beetle_hp), 0, 0],
        DeityDeath => [deity(), i64::from(s.tavern_spells_cast), 0],
        Magicfin => [
            i64::from(s.battlecries_triggered),
            i64::from(s.gold_spent_total) + i64::from(p.gold) + i64::from(p.gold_next_turn),
            i64::from(p.free_refreshes) + i64::from(p.max_gold_bonus) + p.hand.len() as i64,
        ],
        MechSpell => [i64::from(s.magnetized), i64::from(s.spells_cast), 0],
        PirateGold => [
            i64::from(s.gold_spent_total),
            i64::from(s.golden_played),
            i64::from(s.spells_cast),
        ],
        Barrage => [shop(), i64::from(s.gem_atk) + i64::from(s.gem_hp), 0],
        Knight => [i64::from(s.eternal_knight_deaths), 0, 0],
        HandGems => [
            p.board.iter().map(|m| i64::from(m.gems)).sum(),
            i64::from(s.gem_atk) + i64::from(s.gem_hp),
            0,
        ],
        Sludge => [
            i64::from(s.cards_discarded),
            i64::from(s.spell_atk) + i64::from(s.spell_hp),
            0,
        ],
        DragonSpell => [
            i64::from(s.spell_atk) + i64::from(s.spell_hp),
            i64::from(s.tavern_spells_cast),
            0,
        ],
        Volumizer => [magnets(), i64::from(s.magnetized), 0],
        Tidecaller => [hand(), i64::from(s.spells_cast), 0],
        DragonCry | Keywords => [i64::from(s.battlecries_triggered), 0, 0],
        BeastSummon | Poison => [i64::from(s.deathrattles_triggered), 0, 0],
        Shield => [i64::from(s.tavern_spells_cast), 0, 0],
        _ => [0; 3],
    }
}

impl Witness {
    fn before(route: Route, p: &Player, action: Action) -> Self {
        let spell = match action {
            Action::Play { hand, .. } => p.hand.get(hand).and_then(|h| match h.item {
                Item::Spell { card, .. } => Some(card),
                _ => None,
            }),
            Action::Choose(_) => match p.pending.last() {
                Some(Pending::Target {
                    cont: Continuation::HandSpell { hid, .. },
                    ..
                })
                | Some(Pending::ChooseOne {
                    cont: Continuation::HandSpell { hid, .. },
                    ..
                }) => p.hand.iter().find(|h| h.hid == *hid).map(|h| h.item.card()),
                _ => None,
            },
            _ => None,
        };
        let mut others = [(0, 0); 7];
        for (s, m) in others.iter_mut().zip(&p.board) {
            if route != Shield || m.has(Kw::DIVINE_SHIELD) {
                *s = (m.eid, strength(m));
            }
        }
        Self {
            route,
            action,
            spell,
            metric: stat(route, p),
            anchor: anchor(route, p),
            others,
        }
    }

    fn gain(&self, p: &Player, combat: bool) -> i32 {
        let mt = stat(self.route, p);
        let diff = |i: usize| (mt[i] - self.metric[i]).clamp(0, 10_000_000) as i32;
        let current = self.anchor.and_then(|(eid, counter, value)| {
            p.board.iter().find(|m| m.eid == eid).map(|m| {
                (
                    m.counter - counter,
                    (strength(m) - value).clamp(0, 10_000_000) as i32,
                )
            })
        });
        let counter = current.map_or(0, |s| s.0);
        let growth = current.map_or(0, |s| s.1);
        let others = || {
            self.others
                .iter()
                .filter_map(|&(eid, prior)| {
                    p.board
                        .iter()
                        .find(|m| m.eid == eid)
                        .map(|m| (strength(m) - prior).clamp(0, 100_000) as i32)
                })
                .sum::<i32>()
        };
        let play = matches!(self.action, Action::Play { .. } | Action::Choose(_));
        let cast = self.spell.is_some();
        match self.route {
            Keywords | DragonCry if diff(0) >= 1 => others(),
            Lobster | UndeadAttack | Beetle | Knight | DeityDeath => diff(0),
            DemonShop | DemonDeath => {
                diff(0)
                    + if self
                        .spell
                        .is_some_and(|c| matches!(key(c), "BG28_607" | "BG36_880"))
                        && mt[1] < self.metric[1]
                    {
                        4
                    } else {
                        0
                    }
            }
            Behemoth if combat => counter.max(0),
            Tempest if play && counter < 0 => growth,
            Ultraviolet | Hope => counter.max(0),
            Magicfin
                if diff(0) > 0
                    && (diff(1) > 0
                        || diff(2) > 0
                        || self.action == Action::EndTurn
                            && has_key(&p.last_board, APPRENTICE)) =>
            {
                diff(0)
            }
            MechSpell if cast && diff(1) > 0 => diff(0),
            Poison if combat && p.stats.combats_won + p.stats.combats_lost > 0 => 1,
            PirateGold => {
                if diff(1) > 0 {
                    3 * diff(1)
                } else if self.spell.is_some_and(is_bounty) && diff(2) > 0 {
                    2
                } else {
                    0
                }
            }
            Barrage if self.action == Action::Refresh => diff(0),
            HandGems if self.spell == Some(cardid!("BG20_GEM")) => diff(0),
            Sludge => diff(0) + diff(1),
            DragonSpell => {
                if play || combat {
                    diff(0) + diff(1)
                } else {
                    0
                }
            }
            Volumizer => diff(0) + i32::from(self.action == Action::EndTurn && others() > 0),
            Tidecaller if cast && diff(1) > 0 => diff(0) + others(),
            BeastSummon if combat || play => diff(0),
            Shield if cast && diff(0) > 0 => others(),
            _ => 0,
        }
    }
}

fn threshold(r: Route) -> i32 {
    match r {
        Keywords | DragonCry | Tempest | Shield | Tidecaller => 14,
        Lobster | MechSpell | Ultraviolet | Knight | DragonSpell | Volumizer => 2,
        UndeadAttack | Beetle | DeityDeath => 5,
        DemonShop | DemonDeath | PirateGold | Barrage | HandGems | Sludge | BeastSummon => 3,
        Magicfin | Behemoth | Hope | Poison => 1,
    }
}

impl Log {
    pub fn begin(&mut self, r: Route, game: &Game, seat: usize, a: Action) -> Option<Witness> {
        let p = &game.players[seat];
        let entry = &mut self.routes[r.index()];
        entry.selected.get_or_insert(game.turn);
        if a == Action::EndTurn {
            self.entered = Some(r);
            if r == PirateGold && r.core(p, &p.board) && p.stats.gold_spent_turn >= 15 {
                entry.high_spend = true;
            }
            self.combat_witness = r.core(p, &p.board).then(|| Witness::before(r, p, a));
            return None;
        }
        r.core(p, &p.board).then(|| Witness::before(r, p, a))
    }
    pub fn stop(&mut self) {
        self.entered = None;
        self.combat_witness = None;
    }
    pub fn finish(&mut self, w: Witness, p: &Player) {
        let gain = w.gain(p, false);
        self.routes[w.route.index()].gain = self.routes[w.route.index()].gain.saturating_add(gain);
    }
    pub fn after_combat(&mut self, game: &Game, seat: usize, turn: u32) {
        if self.last_combat >= turn
            || !game
                .last_combats
                .iter()
                .any(|c| c.turn == turn && (c.a == seat || c.b == Some(seat)))
        {
            return;
        }
        self.last_combat = turn;
        for entry in &mut self.routes {
            entry.final_full = false;
        }
        let Some(r) = self.entered.take() else {
            return;
        };
        let p = &game.players[seat];
        let board = &p.last_board;
        let stage = &mut self.routes[r.index()];
        if let Some(w) = self.combat_witness.take() {
            stage.gain = stage.gain.saturating_add(w.gain(p, true));
        }
        if !r.core(p, board) {
            return;
        }
        stage.core.get_or_insert(turn);
        let full = r.full(p, board);
        stage.final_full = full;
        if full {
            stage.full.get_or_insert(turn);
        }
        let working = stage.gain >= threshold(r) && (r != PirateGold || stage.high_spend);
        if working {
            stage.started.get_or_insert(turn);
        }
        if working && full {
            stage.started_full.get_or_insert(turn);
        }
        if !working && !full {
            return;
        }
        let attack = board.iter().map(|m| i64::from(m.atk)).sum();
        let health = board.iter().map(|m| i64::from(m.hp)).sum();
        if stage.best.as_ref().is_some_and(|b| {
            (b.started && b.full, b.started, b.full, b.attack + b.health)
                >= (working && full, working, full, attack + health)
        }) {
            return;
        }
        stage.best = Some(Snapshot {
            turn,
            tavern: p.tier,
            hero: p.hero.map(|c| db().get(c).name_zh.as_str()),
            full,
            started: working,
            attack,
            health,
            board: board
                .iter()
                .map(|m| {
                    let card = db().get(m.card);
                    MinionRow {
                        name: &card.name_zh,
                        card: &card.id,
                        golden: m.golden,
                        attack: m.atk,
                        health: m.hp,
                    }
                })
                .collect(),
        });
    }
}
