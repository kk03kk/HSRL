use crate::cards::{CardId, db};
use crate::script::{TargetSpec, script};
use crate::state::{Continuation, Item, Minion, Pending, Player};
use crate::types::{Kw, Races};
use crate::{Action, Game, cardid};

use super::guide_bot::Plan as BuildPlan;

pub const GATE: u8 = 1;
pub const BALINDA: u8 = 2;
pub const PUPPETEER: u8 = 4;
pub const SURVEYOR: u8 = 8;
pub const BRANN: u8 = 16;
pub const GUIDE_FOUR: u8 = GATE | BALINDA | SURVEYOR | BRANN;

fn part(m: &Minion) -> u8 {
    match db().get(db().normal_of(m.card)).id.as_str() {
        "BG36_640" => GATE,
        "BG35_883" => BALINDA,
        "BG36_104" => PUPPETEER,
        "BG30_121" => SURVEYOR,
        "BG_LOE_077" => BRANN,
        _ => 0,
    }
}

pub fn pieces(board: &[Minion]) -> u8 {
    board.iter().fold(0, |mask, m| mask | part(m))
}

pub fn has_core(mask: u8) -> bool {
    mask & GATE != 0 && mask & (BALINDA | PUPPETEER) != 0
}

pub fn evidence(player: &Player) -> i32 {
    let board = pieces(&player.board);
    let hand = player
        .hand
        .iter()
        .filter_map(|h| h.item.as_minion())
        .fold(0, |mask, m| mask | part(m));
    let weighted = [
        (GATE, 21),
        (BALINDA, 5),
        (PUPPETEER, 5),
        (SURVEYOR, 3),
        (BRANN, 2),
    ];
    let base: i32 = weighted
        .iter()
        .map(|(piece, value)| {
            if board & piece != 0 {
                *value
            } else if hand & piece != 0 {
                *value * 2 / 3
            } else {
                0
            }
        })
        .sum();
    let offered = (board | hand) & GATE == 0
        && player
            .shop
            .iter()
            .filter_map(|slot| slot.item.as_minion())
            .any(|m| part(m) == GATE);
    let can_invest = player.tier >= 5
        && player.health + player.armor >= 18
        && player.board.len() >= 5
        && player.stats.last_damage >= -6;
    let generators = player
        .board
        .iter()
        .filter(|m| {
            m.is_card_id(cardid!("BG36_370"))
                || m.is_card_id(cardid!("BG32_821"))
                || m.is_card_id(cardid!("BG32_111"))
        })
        .count()
        .min(2) as i32;
    base + (player.stats.spell_hp / 3).clamp(0, 7)
        + if offered { 16 } else { 0 }
        + if can_invest { 3 + 3 * generators } else { 0 }
}

fn tea_set() -> CardId {
    cardid!("BG28_888")
}

fn gem() -> CardId {
    cardid!("BG20_GEM")
}

// Positive targeted spells. Avoid transformations, selling effects and stat setters.
fn feeds_gate(card: CardId) -> bool {
    matches!(
        db().get(card).id.as_str(),
        "BG20_GEM"
            | "BG28_503"
            | "BG28_507"
            | "BG28_518"
            | "BG28_520"
            | "BG28_607"
            | "BG28_698"
            | "BG28_825"
            | "BG28_845"
            | "BG28_897"
            | "BG31_880"
            | "BG31_886"
            | "BG32_337"
            | "BG35_911"
            | "BG35_912"
            | "BG36_624"
            | "BG36_880"
            | "BG36_883"
    )
}

fn battlecry_supply(m: &Minion) -> f64 {
    match db().get(db().normal_of(m.card)).id.as_str() {
        "BG20_100" | "BG35_881" => 5.0, // Gems or Arcane Absorptions.
        "BG32_111" => 4.0,              // Tea Sets from Nightmare Par-tea Guest.
        _ => 0.0,
    }
}

pub struct Plan<'a> {
    player: &'a Player,
    turn: u32,
    board: u8,
    healthy: bool,
}

impl<'a> Plan<'a> {
    pub fn new(player: &'a Player, turn: u32) -> Self {
        Self {
            player,
            turn,
            board: pieces(&player.board),
            healthy: player.health + player.armor >= 18,
        }
    }

    fn active(&self) -> bool {
        self.board & GATE != 0
    }

    fn component_value(&self, piece: u8) -> f64 {
        match piece {
            GATE => 40.0,
            BALINDA => {
                if self.active() {
                    32.0
                } else {
                    12.0
                }
            }
            PUPPETEER => {
                if self.active() {
                    26.0
                } else {
                    18.0
                }
            }
            SURVEYOR => {
                if self.active() {
                    21.0
                } else {
                    6.0
                }
            }
            BRANN => {
                if self.active() {
                    15.0
                } else {
                    7.0
                }
            }
            _ => 0.0,
        }
    }

    fn has_gate_waiting(&self) -> bool {
        self.player
            .hand
            .iter()
            .any(|h| h.item.as_minion().is_some_and(|m| part(m) == GATE))
            || self
                .player
                .shop
                .iter()
                .any(|s| s.item.as_minion().is_some_and(|m| part(m) == GATE))
    }

    fn spare_type_value(&self, minion: &Minion) -> f64 {
        if minion.races.is_none() {
            return 0.0;
        }
        if minion.races == Races::ALL {
            return 4.0;
        }
        let others = self
            .player
            .board
            .iter()
            .filter(|m| m.eid != minion.eid && m.races != Races::ALL)
            .fold(Races::NONE, |races, m| races.union(m.races));
        let distinct = minion.races.iter().any(|race| !others.intersects(race));
        1.5 + if distinct { 3.5 } else { 0.0 }
    }
}

impl BuildPlan for Plan<'_> {
    fn protects(&self, m: &Minion) -> bool {
        let piece = part(m);
        self.active()
            && matches!(piece, GATE | BALINDA)
            && !self.player.board.iter().any(|other| {
                part(other) == piece
                    && (other.golden, other.hp, other.eid) > (m.golden, m.hp, m.eid)
            })
    }

    fn keep_bonus(&self, m: &Minion) -> f64 {
        let piece = part(m);
        let duplicates = matches!(piece, GATE | BALINDA | BRANN)
            && self.player.board.iter().any(|other| {
                part(other) == piece
                    && (other.golden, other.hp, other.eid) > (m.golden, m.hp, m.eid)
            });
        let mut value = self.component_value(piece) * if duplicates { 0.26 } else { 1.0 };
        if self.active() || self.player.stats.spell_hp >= 4 {
            value += self.spare_type_value(m);
        }
        if m.is_card_id(cardid!("BG32_821")) {
            value += if self.active() { 13.0 } else { 7.0 };
        }
        if m.is_card_id(cardid!("BG32_111")) {
            value += 6.5;
        }
        if m.is_card_id(cardid!("BG36_370")) {
            value += if self.active() { 26.0 } else { 8.0 };
        }
        if m.is_card_id(cardid!("BG36_701"))
            && self.player.board.iter().any(|b| battlecry_supply(b) > 0.0)
        {
            value += if self.active() { 12.0 } else { 4.0 };
        }
        value
    }

    fn acquisition_bonus(&self, m: &Minion, copies: usize) -> f64 {
        let piece = part(m);
        let unlock = self.player.tier == 5
            && !self.active()
            && self.healthy
            && db().get(m.card).golden.is_some();
        let triple = if unlock {
            match copies {
                2 => 19.0,
                1 => 4.5,
                _ => 0.0,
            }
        } else {
            0.0
        };
        if piece == 0 {
            return triple
                + if self.active() {
                    battlecry_supply(m) * 1.7
                } else {
                    0.0
                };
        }
        triple
            + if self.board & piece == 0 {
                if piece == GATE {
                    13.0
                } else if self.active() {
                    5.0
                } else {
                    0.0
                }
            } else if copies == 2 {
                6.0
            } else if matches!(piece, GATE | BALINDA | BRANN) {
                -0.7 * self.component_value(piece)
            } else {
                0.0
            }
    }

    fn spell_bonus(&self, card: CardId) -> f64 {
        if card == tea_set() {
            let typed = self
                .player
                .board
                .iter()
                .filter(|m| !m.races.is_none())
                .count()
                .min(6);
            return (typed as f64 - 1.0).max(0.0)
                * (1.8 + f64::from(self.player.stats.spell_hp.min(40)) * 0.13);
        }
        if self.active() && feeds_gate(card) {
            return 7.0 + if has_core(self.board) { 5.0 } else { 0.0 };
        }
        if !self.active() && self.player.tier == 6 && card == cardid!("BG31_890") {
            return 11.0;
        }
        if self.active()
            && matches!(
                db().get(card).id.as_str(),
                "BG28_810" | "BG28_800" | "BG28_827" | "BG36_303"
            )
        {
            return 5.0;
        }
        0.0
    }

    fn hand_priority(&self, item: &Item) -> f64 {
        match item {
            Item::Minion(m) if part(m) != 0 || m.is_card_id(cardid!("BG36_370")) => 65.0,
            Item::Spell { card, .. } if self.active() && feeds_gate(*card) => 15.0,
            _ => 0.0,
        }
    }

    fn can_cast(&self, card: CardId) -> bool {
        if self.active()
            && matches!(
                db().get(card).id.as_str(),
                "BG30_804" | "BG28_838" | "EBG_Spell_032"
            )
        {
            return false;
        }
        // Save cheap gems briefly if the Gatekeeper can be deployed this turn.
        card != gem()
            || self.active()
            || !self.healthy
            || !self.has_gate_waiting()
            || self.player.hand.len() >= 7
    }

    fn activate(&self, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        if !self.active() || self.player.hand.len() > 4 {
            return None;
        }
        self.player
            .board
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                let action = Action::Activate(i);
                if !legal(action) {
                    return None;
                }
                if m.is_card_id(cardid!("BG36_370")) {
                    Some((action, 2))
                } else if m.is_card_id(cardid!("BG36_701"))
                    && self.player.board.iter().any(|b| battlecry_supply(b) > 0.0)
                {
                    Some((action, 1))
                } else {
                    None
                }
            })
            .max_by_key(|(_, priority)| *priority)
            .map(|(action, _)| action)
    }

    fn pending(&self, menu: &Pending, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        match menu {
            Pending::ChooseOne { card, .. } if *card == cardid!("BG31_890") && !self.active() => {
                legal(Action::Choose(0)).then_some(Action::Choose(0))
            }
            Pending::ChooseOne { card, .. } if *card == cardid!("BG31_886") && self.active() => {
                legal(Action::Choose(0)).then_some(Action::Choose(0)) // Targeted half of Forest's Bounty.
            }
            Pending::Target {
                options,
                cont: Continuation::Activate { eid },
            } if self.active()
                && self
                    .player
                    .board
                    .iter()
                    .any(|m| m.eid == *eid && m.is_card_id(cardid!("BG36_701"))) =>
            {
                options
                    .iter()
                    .enumerate()
                    .filter_map(|(i, target)| {
                        let m = self.player.board.iter().find(|m| m.eid == *target)?;
                        let supply = battlecry_supply(m) * if m.golden { 2.0 } else { 1.0 };
                        (supply > 0.0 && legal(Action::Choose(i))).then_some((i, supply))
                    })
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| Action::Choose(i))
            }
            Pending::Target {
                options,
                cont: Continuation::HandSpell { hid, .. },
            } if self.active() => {
                let card = self.player.hand.iter().find(|h| h.hid == *hid)?.item.card();
                if feeds_gate(card) {
                    options
                        .iter()
                        .enumerate()
                        .filter_map(|(i, target)| {
                            let gate = self
                                .player
                                .board
                                .iter()
                                .find(|m| m.eid == *target && part(m) == GATE)?;
                            (legal(Action::Choose(i))
                                && (card != cardid!("BG28_507") || !gate.has(Kw::DIVINE_SHIELD)))
                            .then_some((i, (gate.golden, gate.hp)))
                        })
                        .max_by_key(|(_, value)| *value)
                        .map(|(i, _)| Action::Choose(i))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn upgrade(&self, game: &Game, seat: usize, ordinary: bool) -> bool {
        if self.player.tier != 5 || self.active() {
            return ordinary;
        }
        let surplus = self.player.gold - game.upgrade_price(seat);
        let ready = self.turn >= 10
            && self.healthy
            && self.player.board.len() >= 5
            && (surplus >= game.minion_price(seat) || self.player.stats.last_damage >= 0);
        ordinary || ready
    }

    fn search_missing(&self, game: &Game, seat: usize) -> bool {
        self.player.tier == 6
            && self.healthy
            && !has_core(self.board)
            && self.player.actions_this_turn < 38
            && self.player.gold >= game.refresh_price(seat)
    }

    fn attack_bonus(&self, m: &Minion) -> f64 {
        if part(m) == PUPPETEER || m.is_card_id(cardid!("BG32_111")) {
            5.0
        } else {
            0.0
        }
    }
}

pub fn aimed_at_gate(player: &Player, options: &[u32], index: usize, hid: u32) -> bool {
    options.get(index).is_some_and(|target| {
        player
            .board
            .iter()
            .any(|m| m.eid == *target && part(m) == GATE)
    }) && player
        .hand
        .iter()
        .find(|h| h.hid == hid)
        .is_some_and(|hand| {
            let card = hand.item.card();
            card != tea_set()
                && matches!(hand.item, Item::Spell { .. })
                && script(card).is_some_and(|s| s.spell_target != TargetSpec::None)
        })
}
