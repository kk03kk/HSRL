use crate::cards::{CardId, db};
use crate::script::{TargetSpec, script};
use crate::state::{Continuation, Item, Minion, Pending, Player};
use crate::types::{Kw, Races};
use crate::{Action, Game, cardid};

use super::guide_bot::Plan as BuildPlan;

pub const ASHEN: u8 = 1;
pub const ESCAPIST: u8 = 2;
pub const DISTRACTOR: u8 = 4;
const BALINDA: u8 = 8;
const BRANN: u8 = 16;
const WEAVER: u8 = 32;
pub const REWINDER: u8 = 64;
pub const SIX: u8 = ASHEN | ESCAPIST | DISTRACTOR | BALINDA | BRANN | WEAVER;

pub fn part(card: CardId) -> u8 {
    match db().get(db().normal_of(card)).id.as_str() {
        "BG32_873" => ASHEN,
        "BG36_733" => ESCAPIST,
        "BG36_762" => DISTRACTOR,
        "BG35_883" => BALINDA,
        "BG_LOE_077" => BRANN,
        "BGS_004" => WEAVER,
        "BG26_174" => REWINDER,
        _ => 0,
    }
}

pub fn pieces(board: &[Minion]) -> u8 {
    board.iter().fold(0, |mask, m| mask | part(m.card))
}

pub fn has_rewind(mask: u8) -> bool {
    mask & (ASHEN | REWINDER) != 0
}

pub fn has_engine(mask: u8) -> bool {
    mask & (ASHEN | ESCAPIST | DISTRACTOR) == ASHEN | ESCAPIST | DISTRACTOR
}

pub fn needs_six(board: &[Minion]) -> bool {
    pieces(board) & (BALINDA | ESCAPIST) != BALINDA | ESCAPIST
}

pub fn cupcakes() -> CardId {
    cardid!("BG28_607")
}

pub fn cupcake_target(
    player: &Player,
    menu: &Pending,
    legal: &dyn Fn(Action) -> bool,
) -> Option<Action> {
    let Pending::Target {
        options,
        cont: Continuation::HandSpell { hid, .. },
    } = menu
    else {
        return None;
    };
    if !player
        .hand
        .iter()
        .any(|h| h.hid == *hid && h.item.card() == cupcakes())
    {
        return None;
    }
    options
        .iter()
        .enumerate()
        .filter(|(i, _)| legal(Action::Choose(*i)))
        .filter_map(|(i, target)| {
            let m = player
                .board
                .iter()
                .find(|m| m.eid == *target && part(m.card) == DISTRACTOR)?;
            Some((
                i,
                (
                    m.golden,
                    m.has(Kw::DIVINE_SHIELD),
                    m.atk.saturating_add(m.hp),
                ),
            ))
        })
        .max_by_key(|(_, value)| *value)
        .map(|(i, _)| Action::Choose(i))
}

fn malchezaar(m: &Minion) -> bool {
    m.is_card_id(cardid!("BG26_524"))
}

fn percussionist(m: &Minion) -> bool {
    m.is_card_id(cardid!("BG26_525"))
}

pub fn evidence(player: &Player, turn: u32) -> i32 {
    let board = pieces(&player.board);
    let hand = player
        .hand
        .iter()
        .filter_map(|h| h.item.as_minion())
        .fold(0, |mask, m| mask | part(m.card));
    let offered = player
        .shop
        .iter()
        .filter_map(|s| s.item.as_minion())
        .fold(0, |mask, m| mask | part(m.card));
    let owned = board | hand;
    let supply = owned & WEAVER != 0 || player.board.iter().any(malchezaar);
    let early_rewind = turn <= 8
        && (owned | offered) & REWINDER != 0
        && (supply || offered & WEAVER != 0 || turn <= 5);
    let available = owned | offered;
    let late_route = available & ASHEN != 0 && (supply || available & (ESCAPIST | DISTRACTOR) != 0);
    if !early_rewind && !late_route && !has_engine(owned) {
        return 0;
    }
    let weight = [
        (ASHEN, 13),
        (ESCAPIST, 15),
        (DISTRACTOR, 12),
        (WEAVER, 7),
        (BALINDA, 5),
        (BRANN, 4),
    ];
    let owned_score: i32 = weight
        .into_iter()
        .map(|(flag, value)| {
            if board & flag != 0 {
                value
            } else if hand & flag != 0 {
                value * 3 / 4
            } else if offered & flag != 0 && matches!(flag, ASHEN | ESCAPIST | DISTRACTOR | WEAVER)
            {
                value / 2
            } else {
                0
            }
        })
        .sum();
    owned_score
        + if early_rewind {
            if supply { 14 } else { 13 }
        } else {
            0
        }
        + if late_route && !early_rewind { 7 } else { 0 }
        + if has_engine(board) { 9 } else { 0 }
}

pub struct Plan<'a> {
    player: &'a Player,
    board: u8,
    turn: u32,
}

impl<'a> Plan<'a> {
    pub fn new(player: &'a Player, turn: u32) -> Self {
        Self {
            player,
            board: pieces(&player.board),
            turn,
        }
    }

    fn life(&self) -> i32 {
        self.player.health + self.player.armor
    }

    fn donors(&self) -> usize {
        self.player
            .shop
            .iter()
            .filter(|slot| slot.item.as_minion().is_some())
            .count()
    }

    fn weaver_damage(&self) -> i32 {
        self.player
            .board
            .iter()
            .filter(|m| part(m.card) == WEAVER)
            .map(|m| if m.golden { 2 } else { 1 })
            .sum()
    }

    fn key_in_shop(&self) -> bool {
        self.player.gold >= 3
            && self
                .player
                .shop
                .iter()
                .filter_map(|s| s.item.as_minion())
                .any(|m| self.priority_piece(m))
    }

    pub fn priority_piece(&self, m: &Minion) -> bool {
        let key = part(m.card);
        self.board & key == 0
            && (matches!(key, ASHEN | ESCAPIST | DISTRACTOR)
                || key == REWINDER && !has_rewind(self.board)
                || key == WEAVER && has_rewind(self.board)
                || matches!(key, BALINDA | BRANN) && self.board & (ESCAPIST | DISTRACTOR) != 0)
    }

    fn first_copy(&self, m: &Minion, key: u8) -> bool {
        !self.player.board.iter().any(|other| {
            part(other.card) == key && (other.golden, other.hp, other.eid) > (m.golden, m.hp, m.eid)
        })
    }

    fn component_value(&self, key: u8) -> f64 {
        match key {
            REWINDER if self.board & ASHEN == 0 => {
                if self.turn <= 8 || self.board & (WEAVER | ESCAPIST) != 0 {
                    17.0
                } else {
                    8.0
                }
            }
            ASHEN => 21.0,
            ESCAPIST => {
                if has_rewind(self.board) {
                    25.0
                } else {
                    12.0
                }
            }
            DISTRACTOR => {
                if self.board & ESCAPIST != 0 {
                    28.0
                } else {
                    16.0
                }
            }
            BALINDA => {
                if self.board & DISTRACTOR != 0 {
                    20.0
                } else {
                    8.0
                }
            }
            BRANN => {
                if self.board & (ASHEN | ESCAPIST) != 0 {
                    16.0
                } else {
                    6.0
                }
            }
            WEAVER => {
                if has_rewind(self.board) {
                    18.0
                } else {
                    3.5
                }
            }
            _ => 0.0,
        }
    }

    fn feeds_distractor(&self, card: CardId) -> bool {
        // Restrict to buffs and consumption. Keep destructive, copying and return-to-hand
        // spells on the shared policy's target rules.
        matches!(
            db().get(card).id.as_str(),
            "BG28_607"
                | "BG20_GEM"
                | "BG28_507"
                | "BG28_897"
                | "BG32_337"
                | "BG36_624"
                | "BG36_883"
        )
    }

    pub fn can_play_minion(&self, m: &Minion) -> bool {
        if has_rewind(self.board) || matches!(part(m.card), ASHEN | REWINDER) {
            return true;
        }
        let cost = self.weaver_damage() * i32::from(m.is(Races::DEMON))
            + if percussionist(m) {
                i32::from(self.player.tier)
                    * if m.golden { 2 } else { 1 }
                    * self
                        .player
                        .board
                        .iter()
                        .filter(|b| part(b.card) == BRANN)
                        .map(|b| if b.golden { 3 } else { 2 })
                        .max()
                        .unwrap_or(1)
            } else {
                0
            };
        cost == 0 || self.life() - cost >= if self.turn <= 7 { 23 } else { 27 }
    }

    pub fn health_refresh(&self, game: &Game, seat: usize) -> bool {
        game.refresh_price(seat) > 0 && game.malchezaar(seat).is_some()
    }

    pub fn wants_health_refresh(&self, game: &Game, seat: usize) -> bool {
        let pays_off = self.player.gold >= game.minion_price(seat)
            || self.player.gold > 0 && self.player.tier >= 4
            || self.board & ESCAPIST != 0
            || self.board & ASHEN != 0
                && self
                    .player
                    .board
                    .iter()
                    .any(|m| m.is_card_id(cardid!("BG26_523")));
        self.health_refresh(game, seat) && has_rewind(self.board) && pays_off
    }

    pub fn replenish(&self, game: &Game, seat: usize) -> bool {
        self.board & DISTRACTOR != 0
            && self.donors() < 3
            && !self.key_in_shop()
            && self
                .player
                .hand
                .iter()
                .any(|h| h.locked == 0 && h.item.card() == cupcakes())
            && (self.wants_health_refresh(game, seat)
                || !self.health_refresh(game, seat)
                    && (game.refresh_price(seat) == 0
                        || self.donors() == 0 && self.player.gold >= game.refresh_price(seat)))
    }
}

impl BuildPlan for Plan<'_> {
    fn protects(&self, m: &Minion) -> bool {
        let key = part(m.card);
        let protected = match key {
            REWINDER => {
                self.board & ASHEN == 0 && (self.turn <= 8 || self.board & (WEAVER | ESCAPIST) != 0)
            }
            ASHEN | ESCAPIST | DISTRACTOR => true,
            WEAVER => has_rewind(self.board) || self.turn <= 5,
            BALINDA | BRANN => self.board & (ASHEN | ESCAPIST | DISTRACTOR) != 0,
            _ => false,
        };
        protected && self.first_copy(m, key)
    }

    fn keep_bonus(&self, m: &Minion) -> f64 {
        let key = part(m.card);
        let component = self.component_value(key)
            * if key != 0 && !self.first_copy(m, key) {
                0.1
            } else {
                1.0
            };
        if malchezaar(m) {
            component + if has_rewind(self.board) { 8.0 } else { 1.0 }
        } else if m.is_card_id(cardid!("BG26_523")) && has_rewind(self.board) {
            component + if self.board & SIX == SIX { 0.0 } else { 15.0 }
        } else {
            component
        }
    }

    fn acquisition_bonus(&self, m: &Minion, copies: usize) -> f64 {
        if !self.can_play_minion(m) {
            return -40.0;
        }
        let key = part(m.card);
        let triple =
            if self.player.tier == 5 && self.board & (BALINDA | ESCAPIST) != BALINDA | ESCAPIST {
                match copies {
                    2 => 13.0,
                    1 => 3.0,
                    _ => 0.0,
                }
            } else {
                0.0
            };
        triple
            + match m {
                _ if self.priority_piece(m) => 10.0,
                _ if key == REWINDER && self.board & ASHEN != 0 => -20.0,
                _ if key != 0 && self.board & key != 0 && copies != 2 => -9.0,
                _ if percussionist(m) && has_rewind(self.board) => {
                    if self.board & (BRANN | ESCAPIST) != 0 {
                        22.0
                    } else {
                        12.0
                    }
                }
                _ if malchezaar(m)
                    && has_rewind(self.board)
                    && !self.player.board.iter().any(malchezaar) =>
                {
                    13.0
                }
                _ if m.is(Races::DEMON) && has_rewind(self.board) && self.board & WEAVER != 0 => {
                    if self.board & ESCAPIST != 0 {
                        7.0
                    } else if self.turn <= 8 {
                        5.0
                    } else {
                        2.0
                    }
                }
                _ => 0.0,
            }
    }

    fn spell_bonus(&self, card: CardId) -> f64 {
        if self.board & DISTRACTOR != 0 && self.feeds_distractor(card) {
            if self.board & BALINDA != 0 { 12.0 } else { 6.0 }
        } else {
            0.0
        }
    }

    fn hand_priority(&self, item: &Item) -> f64 {
        match item {
            Item::Minion(m) if self.priority_piece(m) => match part(m.card) {
                ASHEN | REWINDER => 250.0,
                ESCAPIST | WEAVER => 175.0,
                DISTRACTOR | BALINDA | BRANN => 145.0,
                _ => 0.0,
            },
            Item::Spell { card, .. } if *card == cupcakes() && self.board & DISTRACTOR != 0 => 65.0,
            Item::Minion(m)
                if percussionist(m)
                    && has_rewind(self.board)
                    && self.board & ESCAPIST != 0
                    && self.player.hand.len() <= 6 =>
            {
                46.0
            }
            _ => 0.0,
        }
    }

    fn can_cast(&self, card: CardId) -> bool {
        card != cupcakes()
            || if self.board & DISTRACTOR != 0 {
                self.player.hand.len() >= 8
                    || !self.key_in_shop() && (self.donors() > 0 || self.life() <= 15)
            } else {
                self.player.hand.len() >= 7 || self.life() <= 14
            }
    }

    fn activate(&self, _legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        None
    }

    fn pending(&self, menu: &Pending, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        if let Some(action) = cupcake_target(self.player, menu, legal) {
            return Some(action);
        }
        if let Pending::Discover { source, options } = menu
            && self.board & ESCAPIST != 0
            && has_rewind(self.board)
            && db().normal_of(*source) == cardid!("BG26_525")
        {
            let value = |m: &Minion| {
                self.acquisition_bonus(m, 0)
                    + 2.5 * f64::from(m.tier)
                    + f64::from((m.atk + m.hp).max(0)).sqrt()
            };
            return options
                .iter()
                .enumerate()
                .filter(|(i, _)| legal(Action::Choose(*i)))
                .filter_map(|(i, item)| item.as_minion().map(|m| (i, value(m))))
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| Action::Choose(i));
        }
        let Pending::Target {
            options,
            cont: Continuation::HandSpell { hid, .. },
        } = menu
        else {
            return None;
        };
        let card = self.player.hand.iter().find(|h| h.hid == *hid)?.item.card();
        if !self.feeds_distractor(card)
            || !script(card).is_some_and(|s| s.spell_target != TargetSpec::None)
        {
            return None;
        }
        options
            .iter()
            .enumerate()
            .filter(|(i, _)| legal(Action::Choose(*i)))
            .filter_map(|(i, id)| {
                let m = self.player.board.iter().find(|m| m.eid == *id)?;
                let key = part(m.card);
                (key == DISTRACTOR || card == cupcakes()).then_some((
                    i,
                    (
                        key == DISTRACTOR,
                        m.golden && key == DISTRACTOR,
                        m.has(Kw::DIVINE_SHIELD),
                        m.atk.saturating_add(m.hp),
                    ),
                ))
            })
            .max_by_key(|(_, value)| *value)
            .map(|(i, _)| Action::Choose(i))
    }

    fn upgrade(&self, _game: &Game, _seat: usize, ordinary: bool) -> bool {
        ordinary
    }

    fn search_missing(&self, _game: &Game, _seat: usize) -> bool {
        false
    }
}
