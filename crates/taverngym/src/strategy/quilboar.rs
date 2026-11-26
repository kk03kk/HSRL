use crate::cards::{CardId, db};
use crate::script::script;
use crate::state::{Continuation, Item, Minion, Pending, Player};
use crate::types::{Kw, Races};
use crate::{Action, Game, cardid};

use super::guide_bot::Plan as BuildPlan;

pub const RIDER: u8 = 1;
pub const RAT: u8 = 2;
pub const TUNNELER: u8 = 4;
const REFINER: u8 = 8;
const FIGHTER: u8 = 16;
pub const GUIDE_FIVE: u8 = RIDER | RAT | TUNNELER | REFINER | FIGHTER;

fn component(card: CardId) -> u8 {
    match db().get(db().normal_of(card)).id.as_str() {
        "BG31_323" => RIDER,
        "BG31_326" => RAT,
        "BG36_331" => TUNNELER,
        "BG33_885" => REFINER,
        "BG34_684" => FIGHTER,
        _ => 0,
    }
}

pub fn pieces(board: &[Minion]) -> u8 {
    board.iter().fold(0, |mask, m| mask | component(m.card))
}

pub fn has_core(mask: u8) -> bool {
    mask & RIDER != 0 && mask & (RAT | TUNNELER) != 0
}

pub fn is_choose_one(card: CardId) -> bool {
    script(card).is_some_and(|s| s.choose_one)
}

pub fn evidence(player: &Player) -> i32 {
    let board = pieces(&player.board);
    let hand = player
        .hand
        .iter()
        .filter_map(|h| h.item.as_minion())
        .fold(0, |mask, m| mask | component(m.card));
    let offered = (board | hand) & RIDER == 0
        && player
            .shop
            .iter()
            .any(|s| component(s.item.card()) == RIDER);
    let weighted = [
        (RIDER, 21),
        (RAT, 7),
        (TUNNELER, 8),
        (REFINER, 5),
        (FIGHTER, 3),
    ];
    let owned: i32 = weighted
        .into_iter()
        .map(|(piece, value)| {
            if board & piece != 0 {
                value
            } else if hand & piece != 0 {
                value * 2 / 3
            } else {
                0
            }
        })
        .sum();
    let quill_count = player
        .board
        .iter()
        .filter(|m| m.is(Races::QUILBOAR))
        .count();
    let support = if board & (RAT | TUNNELER | REFINER) != 0 {
        quill_count.min(4) as i32
    } else {
        0
    };
    owned
        + support
        + (player.stats.gem_atk + player.stats.gem_hp).clamp(0, 9)
        + if offered { 16 } else { 0 }
}

pub struct Plan<'a> {
    player: &'a Player,
    turn: u32,
    pieces: u8,
    quills: usize,
    safe: bool,
}

impl<'a> Plan<'a> {
    pub fn new(player: &'a Player, turn: u32) -> Self {
        Self {
            player,
            turn,
            pieces: pieces(&player.board),
            quills: player
                .board
                .iter()
                .filter(|m| m.is(Races::QUILBOAR))
                .count(),
            safe: player.health + player.armor >= 17
                && (player.health + player.armor >= 25 || player.stats.last_damage > -8),
        }
    }

    fn active(&self) -> bool {
        self.pieces & RIDER != 0
    }

    fn gem_value(&self) -> f64 {
        2.0 + f64::from(self.player.stats.gem_atk + self.player.stats.gem_hp)
    }

    fn engine_value(&self, part: u8) -> f64 {
        let active = self.active();
        match part {
            RIDER => {
                34.0 + if self.pieces & (RAT | TUNNELER) != 0 {
                    12.0
                } else {
                    0.0
                }
            }
            RAT => {
                if active {
                    26.0
                } else if self.turn <= 8 {
                    12.0
                } else {
                    8.0
                }
            }
            TUNNELER => {
                if active {
                    26.0
                } else {
                    13.0
                }
            }
            REFINER => {
                if active || self.quills >= 3 {
                    23.0
                } else {
                    12.0
                }
            }
            FIGHTER => {
                if self.gem_value() >= 4.0 {
                    13.0
                } else {
                    6.0
                }
            }
            _ => 0.0,
        }
    }

    fn first_producer(&self, m: &Minion) -> bool {
        let part = component(m.card);
        !self.player.board.iter().any(|other| {
            component(other.card) == part
                && (other.golden, other.hp, other.eid) > (m.golden, m.hp, m.eid)
        })
    }

    fn target(
        &self,
        card: CardId,
        options: &[u32],
        legal: &dyn Fn(Action) -> bool,
    ) -> Option<Action> {
        let confiscation = card == cardid!("BG28_698");
        if !confiscation && card != cardid!("BG20_GEM") {
            return None;
        }
        options
            .iter()
            .enumerate()
            .filter_map(|(i, target)| {
                let (slot, unit) = self
                    .player
                    .board
                    .iter()
                    .enumerate()
                    .find(|(_, m)| m.eid == *target)?;
                let action = Action::Choose(i);
                if !legal(action) {
                    return None;
                }
                let mut score = carrier(unit, self.active());
                if confiscation {
                    // Move gems towards a good carrier without stripping a second carry gratuitously.
                    for neighbor in [slot.checked_sub(1), Some(slot + 1)]
                        .into_iter()
                        .flatten()
                        .filter_map(|j| self.player.board.get(j))
                    {
                        let transfer =
                            f64::from(neighbor.gem_atk.max(0) + neighbor.gem_hp.max(0)).sqrt();
                        score += transfer
                            * ((carrier(unit, self.active()) - carrier(neighbor, self.active()))
                                / 8.0)
                                .clamp(-1.0, 1.0);
                    }
                }
                Some((action, score))
            })
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(action, _)| action)
    }
}

impl BuildPlan for Plan<'_> {
    fn protects(&self, m: &Minion) -> bool {
        let part = component(m.card);
        self.active()
            && (part == RIDER || matches!(part, RAT | TUNNELER | REFINER) && self.first_producer(m))
    }

    fn keep_bonus(&self, m: &Minion) -> f64 {
        let part = component(m.card);
        let mut score = self.engine_value(part);
        if matches!(part, RAT | TUNNELER | FIGHTER) && !self.first_producer(m) {
            score *= 0.6;
        }
        if m.is(Races::QUILBOAR) && self.active() {
            score += 5.0 + 0.35 * self.gem_value().min(20.0);
        }
        if m.is_card_id(cardid!("BG31_327"))
            && (self.active() || self.pieces & (RAT | TUNNELER) != 0)
        {
            score += 12.0;
        }
        if m.is_card_id(cardid!("BG23_017")) {
            score += 8.0;
        }
        score
    }

    fn acquisition_bonus(&self, m: &Minion, copies: usize) -> f64 {
        let part = component(m.card);
        let triple = if self.player.tier == 5
            && !self.active()
            && self.safe
            && db().get(m.card).golden.is_some()
        {
            match copies {
                1 => 3.0,
                2 => 15.0,
                _ => 0.0,
            }
        } else {
            0.0
        };
        if part == RIDER {
            return 16.0 + triple;
        }
        if part != 0 && self.pieces & part == 0 {
            return 6.0 + triple;
        }
        if is_choose_one(m.card) && self.active() {
            return 10.0 + triple;
        }
        if m.is_card_id(cardid!("BG30_123")) || m.is_card_id(cardid!("BG31_320")) {
            return triple
                + if self.quills >= 2 || self.turn <= 8 {
                    5.0
                } else {
                    0.0
                };
        }
        triple
    }

    fn spell_bonus(&self, card: CardId) -> f64 {
        if card == cardid!("BG28_698") {
            return 3.0 + self.gem_value().min(18.0) * 0.7;
        }
        if card == cardid!("BG31_892") || is_choose_one(card) {
            return if self.active() {
                10.0 + (self.quills as f64) * 1.4
            } else {
                0.0
            };
        }
        0.0
    }

    fn hand_priority(&self, item: &Item) -> f64 {
        if item
            .as_minion()
            .is_some_and(|m| m.is_card_id(cardid!("BG31_323")))
        {
            return 140.0;
        }
        match db().get(db().normal_of(item.card())).id.as_str() {
            "BG31_327" | "BG31_893" | "BG30_123" | "BG23_017" => 65.0,
            _ if self.active() && is_choose_one(item.card()) => 25.0,
            _ => 0.0,
        }
    }

    fn can_cast(&self, card: CardId) -> bool {
        // Hold only a small stock when the payoff is immediately available in hand.
        if card == cardid!("BG28_838")
            && self.active()
            && self.player.board.iter().all(|m| m.hp > 20)
        {
            return false;
        }
        !is_choose_one(card)
            || self.active()
            || !self.safe
            || self.player.hand.len() >= 7
            || !self
                .player
                .hand
                .iter()
                .any(|h| h.locked == 0 && component(h.item.card()) == RIDER)
    }

    fn activate(&self, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        if self.quills < 2 || self.player.hand.len() > 5 {
            return None;
        }
        self.player.board.iter().enumerate().find_map(|(i, m)| {
            let action = Action::Activate(i);
            (m.is_card_id(cardid!("BG36_370")) && legal(action)).then_some(action)
        })
    }

    fn pending(&self, menu: &Pending, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        match menu {
            Pending::ChooseOne { card, .. } => {
                let option = match db().get(db().normal_of(*card)).id.as_str() {
                    // Gem Day grows the weaker half. When tied, get enough attack to trade.
                    "BG31_893" => usize::from(self.player.stats.gem_atk > self.player.stats.gem_hp),
                    "BG31_320" => {
                        usize::from(self.safe || self.active() || self.player.hand.len() >= 7)
                    }
                    "BG30_123" => {
                        usize::from(!self.safe && !self.active() && self.gem_value() >= 5.0)
                    }
                    "BG36_330" => usize::from(
                        self.active()
                            && self.gem_value() >= 5.0
                            && (self.player.free_refreshes >= 2
                                || !self.safe
                                || self.player.gold < 3)
                            && self.player.hand.len() < 7,
                    ),
                    "BG36_332" => {
                        usize::from(self.safe && self.turn <= 12 && self.player.board.len() >= 4)
                    }
                    "BG36_341" => 0, // Convert the permanent gem value into board power now.
                    "BG31_890" => usize::from(self.player.tier >= 6 && has_core(self.pieces)),
                    "BG31_881" => usize::from(self.safe && self.turn <= 10),
                    _ => return None,
                };
                legal(Action::Choose(option)).then_some(Action::Choose(option))
            }
            Pending::Target {
                options,
                cont: Continuation::HandSpell { hid, .. },
            } => {
                let card = self.player.hand.iter().find(|h| h.hid == *hid)?.item.card();
                self.target(card, options, legal)
            }
            _ => None,
        }
    }

    fn upgrade(&self, _game: &Game, _seat: usize, ordinary: bool) -> bool {
        ordinary // The tested cadence supplies the health and budget guard.
    }

    fn search_missing(&self, game: &Game, seat: usize) -> bool {
        self.player.tier >= 5
            && self.safe
            && !has_core(self.pieces)
            && self.player.gold >= game.refresh_price(seat) + game.minion_price(seat)
    }

    fn attack_bonus(&self, m: &Minion) -> f64 {
        match component(m.card) {
            REFINER if self.gem_value() < 16.0 => 31.0,
            TUNNELER if self.player.hand.len() < 8 => 24.0,
            _ => 0.0,
        }
    }
}

fn carrier(m: &Minion, active: bool) -> f64 {
    0.45 * f64::from(m.atk.max(0) + m.hp.max(0)).sqrt()
        + if m.has(Kw::DIVINE_SHIELD) { 7.0 } else { 0.0 }
        + if m.has(Kw::WINDFURY) || m.has(Kw::CLEAVE) {
            5.0
        } else {
            0.0
        }
        + if m.is_card_id(cardid!("BG28_583")) {
            16.0
        } else {
            0.0
        }
        + if m.is(Races::QUILBOAR) && active {
            3.0
        } else {
            0.0
        }
}
