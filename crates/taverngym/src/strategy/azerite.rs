use crate::actions::Action;
use crate::cards::{CardId, db};
use crate::content::is_tavern_spell;
use crate::state::{Continuation, Item, Minion, Pending, Player};
use crate::types::{Kw, Races};
use crate::{Game, cardid};

use super::guide_bot::Plan as BuildPlan;

pub const AZERITE: u8 = 1;
pub const BRANN: u8 = 2;
const SURFACER: u8 = 4;
const KEEPER: u8 = 8;
const BALINDA: u8 = 16;
pub const CORE: u8 = AZERITE | BRANN;
pub const ENGINE: u8 = CORE | SURFACER | KEEPER;
pub const FULL: u8 = ENGINE | BALINDA;

fn component(m: &Minion) -> u8 {
    let normal = db().normal_of(m.card);
    if normal == cardid!("BG28_707") {
        AZERITE
    } else if normal == cardid!("BG_LOE_077") {
        BRANN
    } else if normal == cardid!("BG35_881") {
        SURFACER
    } else if normal == cardid!("BG36_701") {
        KEEPER
    } else if normal == cardid!("BG35_883") {
        BALINDA
    } else {
        0
    }
}

pub fn pieces(board: &[Minion]) -> u8 {
    board.iter().fold(0, |mask, m| mask | component(m))
}

pub fn absorption() -> CardId {
    cardid!("BG35_911")
}

pub fn evidence(player: &Player) -> i32 {
    let board = pieces(&player.board);
    let hand = player
        .hand
        .iter()
        .filter_map(|h| h.item.as_minion())
        .fold(0, |mask, m| mask | component(m));
    let weighted = [
        (AZERITE, 15),
        (SURFACER, 5),
        (KEEPER, 3),
        (BRANN, 3),
        (BALINDA, 2),
    ];
    let base: i32 = weighted
        .iter()
        .map(|(part, weight)| {
            if board & part != 0 {
                *weight
            } else if hand & part != 0 {
                *weight * 2 / 3
            } else {
                0
            }
        })
        .sum();
    let offered = (board | hand) & AZERITE == 0
        && player
            .shop
            .iter()
            .filter_map(|slot| slot.item.as_minion())
            .any(|m| component(m) == AZERITE);
    base + (player.stats.tavern_elemental_hp / 9).clamp(0, 10) + if offered { 13 } else { 0 }
}

pub struct Plan<'a> {
    player: &'a Player,
    turn: u32,
    board: u8,
    durable: bool,
}

impl<'a> Plan<'a> {
    pub fn new(player: &'a Player, turn: u32) -> Self {
        Self {
            player,
            turn,
            board: pieces(&player.board),
            durable: player.health + player.armor >= 17,
        }
    }

    fn active(&self) -> bool {
        self.board & AZERITE != 0
    }

    fn component_value(&self, part: u8) -> f64 {
        let value = match part {
            AZERITE => 37.0,
            BRANN => 23.0 + if self.active() { 10.0 } else { 0.0 },
            SURFACER => 19.0 + if self.active() { 12.0 } else { 0.0 },
            KEEPER => {
                if self.board & SURFACER != 0 {
                    30.0
                } else if self.active() {
                    18.0
                } else {
                    7.0
                }
            }
            BALINDA => {
                23.0 + if self.board & SURFACER != 0 {
                    13.0
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        if !self.durable && !self.active() && part != AZERITE {
            value * 0.65
        } else {
            value
        }
    }

    fn donor_health(&self) -> i32 {
        self.player
            .shop
            .iter()
            .filter_map(|s| s.item.as_minion())
            .map(|m| m.hp)
            .max()
            .unwrap_or(0)
    }
}

impl BuildPlan for Plan<'_> {
    fn protects(&self, m: &Minion) -> bool {
        let part = component(m);
        self.active()
            && part != 0
            && !self.player.board.iter().any(|other| {
                component(other) == part
                    && (other.golden, other.hp, other.eid) > (m.golden, m.hp, m.eid)
            })
    }

    fn keep_bonus(&self, m: &Minion) -> f64 {
        let part = component(m);
        let value = self.component_value(part);
        if part != 0 {
            // Keep one of each effect. Multiple Azerites remain useful because they stack.
            let redundant = part != AZERITE
                && self.player.board.iter().any(|other| {
                    component(other) == part
                        && (other.golden, other.hp, other.eid) > (m.golden, m.hp, m.eid)
                });
            return value * if redundant { 0.27 } else { 1.0 };
        }
        if self.board & SURFACER != 0 && m.is(Races::ELEMENTAL) {
            return carrier(m).max(0.0) * 1.6;
        }
        if m.is_card_id(cardid!("BG33_823")) {
            return if self.active() { 11.0 } else { 6.0 };
        }
        if m.is_card_id(cardid!("BG33_825"))
            && self
                .player
                .board
                .iter()
                .any(|b| b.is_card_id(cardid!("BG33_823")))
        {
            return 12.0;
        }
        0.0
    }

    fn acquisition_bonus(&self, m: &Minion, copies: usize) -> f64 {
        let part = component(m);
        let reward_tier = match self.player.tier {
            4 if !self.active() || self.board & BRANN == 0 => 1.0,
            5 if self.active() && self.board & BALINDA == 0 => 0.8,
            _ => 0.0,
        };
        let discover = if !m.golden && db().get(m.card).golden.is_some() && self.durable {
            reward_tier
                * match copies {
                    1 => 4.5,
                    2 => 12.0,
                    _ => 0.0,
                }
        } else {
            0.0
        };
        if part == 0 {
            return discover;
        }
        // The ordinary offer score already includes keep value. This term adjusts
        // the premium for new pieces, duplicates and completing a golden piece.
        if self.board & part == 0 {
            let missing = match part {
                KEEPER if self.board & SURFACER == 0 && !self.active() => 0.0,
                BALINDA if self.active() && self.board & SURFACER != 0 => 17.0,
                _ => 7.0,
            };
            return discover + missing;
        }
        if copies == 2 {
            return discover + 8.0;
        }
        if part == AZERITE {
            return if self
                .player
                .board
                .iter()
                .filter(|b| component(b) == AZERITE)
                .count()
                < 2
            {
                1.0
            } else {
                -16.0
            };
        }
        if part == SURFACER || part == KEEPER {
            return -5.0;
        }
        -0.65 * self.component_value(part)
    }

    fn spell_bonus(&self, card: CardId) -> f64 {
        if card == absorption() {
            return 6.0 + 0.22 * f64::from(self.donor_health());
        }
        if self.active() && is_tavern_spell(card) {
            return match db().get(card).id.as_str() {
                "BG28_810" | "BG36_303" => 15.0, // Net-zero gold at printed cost.
                "BG28_800" | "BG28_827" | "BG28_849" => 11.0,
                "BG28_805" if self.turn <= 12 && self.durable => 9.0,
                _ => 4.0,
            };
        }
        0.0
    }

    fn hand_priority(&self, item: &Item) -> f64 {
        match item {
            Item::Minion(m) => match component(m) {
                AZERITE | BRANN | BALINDA => 110.0,
                SURFACER => 45.0,
                KEEPER => 35.0,
                _ => 0.0,
            },
            Item::Spell { card, .. } if *card == absorption() => 12.0,
            Item::Spell { card, .. } if self.active() && is_tavern_spell(*card) => {
                // Resolve shop-swapping effects after cashing in a good donor.
                if *card == cardid!("BG28_849") {
                    1.0
                } else {
                    23.0
                }
            }
            _ => 0.0,
        }
    }

    fn can_cast(&self, card: CardId) -> bool {
        card != absorption()
            || (self.donor_health() > 0
                && (self.active()
                    || self.donor_health() >= 12
                    || !self.durable
                    || self.player.hand.len() >= 7))
    }

    fn activate(&self, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        if self.board & SURFACER == 0 || self.player.hand.len() > 7 || self.donor_health() == 0 {
            return None;
        }
        self.player.board.iter().enumerate().find_map(|(i, m)| {
            let action = Action::Activate(i);
            (component(m) == KEEPER && legal(action)).then_some(action)
        })
    }

    fn pending(&self, menu: &Pending, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        match menu {
            Pending::Trinket { options, .. } => options.iter().enumerate().find_map(|(i, card)| {
                let action = Action::Choose(i);
                (*card == cardid!("BG30_MagicItem_431") && legal(action)).then_some(action)
            }),
            Pending::HeroSelect { options } => options.iter().enumerate().find_map(|(i, card)| {
                let action = Action::Choose(i);
                // Dinotamer Brann gets Brann after buying Battlecry minions.
                (*card == cardid!("TB_BaconShop_HERO_43") && legal(action)).then_some(action)
            }),
            Pending::Target { options, cont } => {
                let kelp = matches!(cont, Continuation::Activate { eid }
                    if self.player.board.iter().any(|m| m.eid == *eid && component(m) == KEEPER));
                let absorb = match cont {
                    Continuation::HandSpell { hid, .. } => self
                        .player
                        .hand
                        .iter()
                        .any(|h| h.hid == *hid && h.item.card() == absorption()),
                    Continuation::Spell { card, .. } => *card == absorption(),
                    _ => false,
                };
                if !kelp && !absorb {
                    return None;
                }
                options
                    .iter()
                    .enumerate()
                    .filter_map(|(i, eid)| {
                        let action = Action::Choose(i);
                        let minion = self.player.board.iter().find(|m| m.eid == *eid)?;
                        if !legal(action) || (kelp && component(minion) != SURFACER) {
                            return None;
                        }
                        Some((
                            action,
                            if kelp {
                                f64::from(u8::from(minion.golden))
                            } else {
                                carrier(minion)
                            },
                        ))
                    })
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(action, _)| action)
            }
            Pending::ChooseOne { card, .. } if *card == cardid!("BG31_890") => {
                // Boundless Potential: first option is a minion of the current tier.
                let choice =
                    Action::Choose(usize::from(self.board == FULL && self.player.tier == 5));
                legal(choice).then_some(choice)
            }
            _ => None,
        }
    }

    fn upgrade(&self, game: &Game, seat: usize, ordinary: bool) -> bool {
        let player = self.player;
        let price = game.upgrade_price(seat);
        let minion_price = game.minion_price(seat);
        if player.tier >= 3
            && player.gold < price + minion_price
            && player
                .shop
                .iter()
                .filter_map(|slot| slot.item.as_minion())
                .any(|m| {
                    let part = component(m);
                    part != 0 && self.board & part == 0 && self.component_value(part) >= 16.0
                })
        {
            return false;
        }
        let health = player.health + player.armor;
        let tempo: i64 = player
            .board
            .iter()
            .map(|m| i64::from(m.atk) + i64::from(m.hp))
            .sum();
        let spare = player.gold >= price + minion_price;
        let stable = health >= 27 && player.stats.last_damage > -8;
        match player.tier {
            3 => {
                ordinary
                    || (self.turn >= 6
                        && player.board.len() >= 4
                        && (spare || stable && tempo >= 26))
            }
            4 => {
                ordinary
                    || (self.turn >= 8
                        && player.board.len() >= 5
                        && (spare || stable && tempo >= 40))
            }
            5 => {
                let can_cash_in = self.active() && (self.board & (SURFACER | BRANN) != 0);
                let ahead = can_cash_in && tempo >= 115 && player.stats.last_damage >= 0;
                (self.board & BALINDA == 0
                    && (self.turn >= 9 && ahead
                        || self.turn >= 10
                            && self.durable
                            && (spare || can_cash_in && tempo >= 65)))
                    || ordinary && (spare || self.turn >= 12 || can_cash_in)
            }
            _ => ordinary,
        }
    }

    fn search_missing(&self, game: &Game, seat: usize) -> bool {
        self.player.actions_this_turn < 42
            && self.player.tier >= 4
            && self.turn >= 7
            && (self.active() || self.durable && self.player.board.len() >= 5)
            && self.board != FULL
            && (self.player.tier >= 5 || self.board & (SURFACER | KEEPER) != (SURFACER | KEEPER))
            && self.player.gold >= game.refresh_price(seat)
    }
}

fn carrier(m: &Minion) -> f64 {
    // The best recipient converts one buff into multiple attacks or absorbs a hit
    // for free. When one carrier is far ahead, give the next buff a second threat.
    let mut value = 0.0;
    for (keyword, bonus) in [
        (Kw::DIVINE_SHIELD, 7.0),
        (Kw::WINDFURY, 5.0),
        (Kw::CLEAVE, 6.0),
    ] {
        if m.has(keyword) {
            value += bonus;
        }
    }
    if m.is_card_id(cardid!("BGS_126")) {
        value += 6.0;
    }
    if m.has(Kw::VENOMOUS) {
        value -= 4.0;
    }
    value - 0.27 * f64::from(m.hp.max(0)).sqrt()
}
