use serde::Serialize;
use taverngym::cards::{CardId, db};
use taverngym::content::{is_bounty, is_tavern_spell};
use taverngym::state::{Continuation, Item, Pending, Player};
use taverngym::types::{Kw, Races};
use taverngym::{Action, Game, cardid};

use super::apm::{self, Route};

#[derive(Default)]
pub struct Formation {
    pub routes: [Progress; 4],
    pub peak_spend: i32,
    pub peak_refund: i32,
    pub bounties_received: u32,
    pub bounties_cast: u32,
    pub wealthy_gold: i32,
    pub loop_turn: Option<u32>,
    pub last: Option<Warband>,
    spend_turn: u32,
    spend: i32,
    refund: i32,
    deity_ready: bool,
}

#[derive(Default)]
pub struct Progress {
    pub selected: bool,
    pub core: Option<u32>,
    pub working: Option<u32>,
    pub full: Option<u32>,
    pub final_full: bool,
    pub gain: i32,
    pub extra: i32,
    pub capstone: i32,
}

struct Payoffs {
    undead_atk: i32,
    shop: usize,
    tavern_growth: bool,
    felboar_stats: i32,
    magnetized: i32,
    attachments: usize,
    boxer_attachments: usize,
    deity: i32,
}

pub struct Witness {
    before: Payoffs,
    gold: i32,
    spent: i32,
    bounties: u32,
    spell: Option<CardId>,
    bought: bool,
    magnetized_onto_other: bool,
    had_fear: bool,
    active: [bool; 4],
}

fn bounty_count(p: &Player) -> u32 {
    p.hand.iter().filter(|h| is_bounty(h.item.card())).count() as u32
}

impl Payoffs {
    fn new(p: &Player) -> Self {
        let minion = |card| p.board.iter().find(|m| m.is_card_id(card));
        Self {
            undead_atk: p.stats.undead_atk,
            shop: p
                .shop
                .iter()
                .filter(|s| s.item.as_minion().is_some())
                .count(),
            tavern_growth: p.stats.tavern_atk > 0
                || p.stats.tavern_hp > 0
                || p.tavern_turn_health > 0
                || p.effects
                    .iter()
                    .any(|e| matches!(db().get(e.card).id.as_str(), "BG34_689" | "BG34_444")),
            felboar_stats: minion(cardid!("BG28_633")).map_or(0, |m| m.atk.saturating_add(m.hp)),
            magnetized: p.stats.magnetized,
            attachments: minion(cardid!("BG36_367")).map_or(0, |m| m.attached.len()),
            boxer_attachments: minion(cardid!("BG26_149")).map_or(0, |m| m.attached.len()),
            deity: p.deity.as_ref().map_or(0, |d| d.atk.saturating_add(d.hp)),
        }
    }
}

impl Witness {
    pub fn before_action(game: &Game, seat: usize, action: Action) -> Option<Self> {
        let p = &game.players[seat];
        if !game.lobby_races.intersects(Races::PIRATE)
            || !apm::economy(&p.board)
            || action == Action::EndTurn
        {
            return None;
        }
        let spell = match action {
            Action::Play { hand, .. } if p.pending.is_empty() => {
                if let Item::Spell { card, .. } = p.hand.get(hand)?.item {
                    Some(card)
                } else {
                    None
                }
            }
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
        Some(Self {
            before: Payoffs::new(p),
            gold: p.gold,
            spent: p.stats.gold_spent_total,
            bounties: bounty_count(p),
            spell,
            bought: matches!(action, Action::Buy(_)),
            magnetized_onto_other: matches!(action, Action::Magnetize { board, .. }
                if p.board.get(board).is_some_and(|m| !m.is_card_id(cardid!("BG26_149")))),
            had_fear: apm::has(&p.board, "BG36_111"),
            active: Route::ALL
                .map(|r| game.lobby_races.contains(r.required()) && r.payoff(&p.board)),
        })
    }

    pub fn record(self, formation: &mut Formation, p: &Player, turn: u32) {
        if formation.spend_turn != turn {
            formation.spend_turn = turn;
            formation.spend = 0;
            formation.refund = 0;
        }
        let spent = (p.stats.gold_spent_total - self.spent).max(0);
        let refunded = p.gold - self.gold + spent;
        formation.spend += spent;
        formation.refund += refunded.max(0);
        formation.peak_spend = formation.peak_spend.max(formation.spend);
        formation.peak_refund = formation.peak_refund.max(formation.refund);
        let is_cast = self.spell.is_some_and(is_bounty);
        formation.bounties_cast += u32::from(is_cast);
        formation.bounties_received +=
            bounty_count(p).saturating_sub(self.bounties.saturating_sub(u32::from(is_cast)));
        if self.spell == Some(cardid!("BG33_815")) {
            formation.wealthy_gold += refunded.max(0);
        }
        if formation.spend >= 15 && formation.refund >= 4 && formation.bounties_cast >= 2 {
            formation.loop_turn.get_or_insert(turn);
        }
        let after = Payoffs::new(p);
        let [undead, tavern, mech, deity] = &mut formation.routes;
        if self.active[Route::Undead.index()] && self.spell.is_some() {
            undead.gain += (after.undead_atk - self.before.undead_atk).max(0);
        }
        if self.active[Route::Tavern.index()]
            && self.spell.is_some()
            && after.shop < self.before.shop
            && after.felboar_stats > self.before.felboar_stats
        {
            tavern.gain += (self.before.shop - after.shop) as i32;
            tavern.extra += i32::from(self.before.tavern_growth);
        }
        if self.active[Route::Mech.index()] {
            mech.gain += (after.magnetized - self.before.magnetized).max(0);
            if self.bought && after.attachments > self.before.attachments {
                mech.extra += 1;
            }
            if self.magnetized_onto_other && after.boxer_attachments > self.before.boxer_attachments
            {
                mech.capstone += 1;
            }
        }
        if self.active[Route::Deity.index()] {
            let gain = (after.deity - self.before.deity).max(0);
            if self.bought {
                deity.gain += gain;
            }
            if self.had_fear && self.spell.is_some_and(is_tavern_spell) && gain > 0 {
                deity.capstone += 1;
            }
        }
    }
}

#[derive(Serialize)]
pub struct Warband {
    pub turn: u32,
    pub tier: u8,
    pub hero: Option<&'static str>,
    pub attack: i64,
    pub health: i64,
    pub undead_attack: i32,
    pub tavern_attack: i32,
    pub tavern_health: i32,
    pub deity_attack: i32,
    pub deity_health: i32,
    pub minions: Vec<MinionRow>,
}

#[derive(Serialize)]
pub struct MinionRow {
    pub name: &'static str,
    pub card: &'static str,
    pub golden: bool,
    pub reborn: bool,
    pub attack: i32,
    pub health: i32,
}

impl Warband {
    pub fn new(p: &Player, turn: u32, tier: u8) -> Self {
        let board = &p.last_board;
        Self {
            turn,
            tier,
            hero: p.hero.map(|id| db().get(id).name_zh.as_str()),
            attack: board.iter().map(|m| i64::from(m.atk)).sum(),
            health: board.iter().map(|m| i64::from(m.hp)).sum(),
            undead_attack: p.stats.undead_atk,
            tavern_attack: p.stats.tavern_atk,
            tavern_health: p.stats.tavern_hp,
            deity_attack: p.deity.as_ref().map_or(0, |d| d.atk),
            deity_health: p.deity.as_ref().map_or(0, |d| d.hp),
            minions: board
                .iter()
                .map(|m| {
                    let d = db().get(m.card);
                    MinionRow {
                        name: &d.name_zh,
                        card: &d.id,
                        golden: m.golden,
                        reborn: m.has(Kw::REBORN),
                        attack: m.atk,
                        health: m.hp,
                    }
                })
                .collect(),
        }
    }
}

impl Formation {
    pub fn record_turn(&mut self, game: &Game, seat: usize, turn: u32, tier: u8) {
        let p = &game.players[seat];
        let board = &p.last_board;
        let economy = apm::economy(board);
        self.deity_ready = false;
        for route in Route::ALL {
            let progress = &mut self.routes[route.index()];
            let eligible = economy && game.lobby_races.contains(route.required());
            progress.final_full = eligible && route.full(board);
            if progress.final_full {
                progress.full.get_or_insert(turn);
            }
            if !eligible || !route.payoff(board) {
                continue;
            }
            progress.core.get_or_insert(turn);
            if self.loop_turn.is_none() {
                continue;
            }
            let working = match route {
                Route::Undead => progress.gain >= 12,
                Route::Tavern => progress.gain >= 3 && progress.extra >= 2,
                Route::Mech => progress.gain >= 6 && progress.extra >= 2,
                Route::Deity => {
                    self.deity_ready = progress.gain >= 72;
                    false
                }
                Route::Golden => unreachable!("tracked by catalog"),
            };
            if working {
                progress.working.get_or_insert(turn);
            }
        }
        self.last = economy.then(|| Warband::new(p, turn, tier));
    }

    pub fn after_combat(&mut self, p: &Player, turn: u32) {
        if self.deity_ready && p.stats.last_deity {
            let deity = &mut self.routes[Route::Deity.index()];
            deity.extra += 1;
            deity.working.get_or_insert(turn);
        }
    }
}
