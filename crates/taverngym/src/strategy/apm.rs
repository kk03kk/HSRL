use crate::cards::{CardId, db};
use crate::content::{is_bounty, is_tavern_spell};
use crate::state::{Continuation, Item, Minion, Pending, Player};
use crate::types::{Kw, Races};
use crate::{Action, Game};

use super::guide_bot::Plan as BuildPlan;
use super::undead_apm;

pub const ROGERS: &str = "BG33_823";
pub const PRIVATEER: &str = "BG33_825";
pub const ESCAPEE: &str = "BG36_523";
const BRANN: &str = "BG_LOE_077";
const CONNIVER: &str = "BG36_369";
const NAVIGATOR: &str = "BG32_231";
use undead_apm::{
    BELLRINGER, BUTCHERING, GRAVEROBBER, HANDLESS, MAW_CASTER, PHANTOM, PROSTHETIC_HAND,
    TOMB_TURNING, WEAVER,
};
const FELBOAR: &str = "BG28_633";
const BALINDA: &str = "BG35_883";
const DISTRACTOR: &str = "BG36_762";
const GEOMANT: &str = "BG36_370";
const REVEILLE: &str = "BG36_367";
const SNAPPER: &str = "BG36_851";
const BEATBOXER: &str = "BG26_149";
const DRONE: &str = "BG36_506";
const DOUBT: &str = "BG36_109";
const FEAR: &str = "BG36_111";
const CONVERTER: &str = "BG36_318";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Undead,
    Tavern,
    Mech,
    Deity,
    Golden,
}

impl Route {
    pub const ALL: [Self; 4] = [Self::Undead, Self::Tavern, Self::Mech, Self::Deity];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Undead => "手速亡灵",
            Self::Tavern => "手速酒馆强化",
            Self::Mech => "手速磁力",
            Self::Deity => "手速神明",
            Self::Golden => "手速金色",
        }
    }

    pub fn required(self) -> Races {
        Races::PIRATE.union(match self {
            Self::Undead => Races::UNDEAD,
            Self::Tavern => Races::QUILBOAR,
            Self::Mech => Races::MECH,
            Self::Deity => Races::ABERRATION,
            Self::Golden => Races::PIRATE,
        })
    }

    fn pieces(self) -> &'static [(&'static str, i32)] {
        match self {
            Self::Undead => &[(WEAVER, 23), (PHANTOM, 14), (HANDLESS, 11)],
            Self::Tavern => &[
                (FELBOAR, 17),
                (GEOMANT, 11),
                (DISTRACTOR, 9),
                (BALINDA, 14),
                (BRANN, 10),
            ],
            Self::Mech => &[(REVEILLE, 22), (SNAPPER, 19), (BEATBOXER, 24), (DRONE, 4)],
            Self::Deity => &[(DOUBT, 23), (FEAR, 23), (CONVERTER, 5)],
            Self::Golden => &[
                ("BG36_344", 25),
                (ESCAPEE, 21),
                ("BG36_524", 9),
                ("BG26_817", 8),
            ],
        }
    }

    fn critical(self, card: &str) -> bool {
        match self {
            Self::Undead => matches!(card, WEAVER | PHANTOM | HANDLESS),
            Self::Tavern => matches!(card, FELBOAR | BALINDA | BRANN),
            Self::Mech => matches!(card, REVEILLE | SNAPPER | BEATBOXER),
            Self::Deity => matches!(card, DOUBT | FEAR),
            Self::Golden => matches!(card, ESCAPEE | "BG36_344"),
        }
    }

    pub fn payoff(self, board: &[Minion]) -> bool {
        has(board, self.pieces()[0].0)
    }

    pub fn full(self, board: &[Minion]) -> bool {
        economy(board)
            && match self {
                Self::Undead => has(board, WEAVER) && has(board, PHANTOM) && has(board, HANDLESS),
                Self::Tavern => has(board, FELBOAR) && has(board, BALINDA) && has(board, BRANN),
                Self::Mech => has(board, REVEILLE) && has(board, SNAPPER) && has(board, BEATBOXER),
                Self::Deity => has(board, DOUBT) && has(board, FEAR),
                Self::Golden => has(board, ESCAPEE) && has(board, "BG36_344"),
            }
    }
}

fn normal(card: CardId) -> &'static str {
    &db().get(db().normal_of(card)).id
}
fn id(m: &Minion) -> &'static str {
    normal(m.card)
}
pub fn has(board: &[Minion], card: &str) -> bool {
    board.iter().any(|m| id(m) == card)
}
pub fn economy(board: &[Minion]) -> bool {
    has(board, ROGERS) && has(board, PRIVATEER)
}

fn presence(player: &Player, card: &str) -> i32 {
    if has(&player.board, card) {
        4
    } else if player
        .hand
        .iter()
        .any(|h| h.item.as_minion().is_some_and(|m| id(m) == card))
    {
        3
    } else if player
        .shop
        .iter()
        .any(|s| s.item.as_minion().is_some_and(|m| id(m) == card))
    {
        2
    } else {
        0
    }
}

pub fn golden_evidence(p: &Player) -> Option<i32> {
    let rogers = presence(p, ROGERS);
    let escapee = presence(p, ESCAPEE);
    let hooktusk = presence(p, "BG36_344");
    let privateer = presence(p, PRIVATEER);
    (rogers + escapee >= 3 && (hooktusk > 0 || privateer > 0 || escapee >= 2)).then_some(
        7 + 4 * (rogers + escapee)
            + 8 * hooktusk
            + 3 * privateer
            + 10 * i32::from(escapee >= 3 && hooktusk >= 3),
    )
}

pub fn candidate(game: &Game, seat: usize, previous: Option<Route>) -> Option<(Route, i32)> {
    let p = &game.players[seat];
    if game.turn < 8 || p.tier < 4 || !game.lobby_races.intersects(Races::PIRATE) {
        return None;
    }
    let rogers = presence(p, ROGERS);
    let privateer = presence(p, PRIVATEER);
    let escapee = presence(p, ESCAPEE);
    if rogers == 0 && escapee == 0 && privateer == 0 {
        return None;
    }
    Route::ALL
        .into_iter()
        .filter(|r| game.lobby_races.contains(r.required()))
        .filter_map(|r| {
            let payoff = r
                .pieces()
                .iter()
                .map(|(card, weight)| presence(p, card) * weight / 4)
                .sum::<i32>();
            if rogers == 0 && escapee == 0 && payoff < 9 {
                return None;
            }
            let continuity =
                i32::from(previous == Some(r)) * if r.payoff(&p.board) { 9 } else { 4 };
            Some((
                r,
                rogers * 12 + privateer * 3 + escapee * 5 + payoff + continuity,
            ))
        })
        .max_by_key(|(r, score)| (*score, std::cmp::Reverse(r.index())))
}

pub struct Plan<'a> {
    p: &'a Player,
    pub route: Route,
    ready: bool,
    rogers: bool,
    undead_specialist: bool,
    undead_finance: bool,
    reborn_conversion: bool,
    advanced: bool,
    board_size: usize,
}

impl<'a> Plan<'a> {
    pub fn new(p: &'a Player, route: Route) -> Self {
        Self {
            p,
            route,
            ready: economy(&p.board),
            rogers: has(&p.board, ROGERS),
            undead_specialist: false,
            undead_finance: false,
            reborn_conversion: false,
            advanced: false,
            board_size: crate::actions::N_BOARD,
        }
    }

    pub fn with_undead_specialist(
        mut self,
        enabled: bool,
        finance: bool,
        board_size: usize,
    ) -> Self {
        self.undead_specialist = enabled && self.route == Route::Undead;
        self.undead_finance = self.undead_specialist && finance;
        self.board_size = board_size;
        self
    }

    pub fn can_play_minion(&self, m: &Minion, turn: u32) -> bool {
        if self.undead_finance && matches!(id(m), GRAVEROBBER | MAW_CASTER) {
            if self.converting_reborn() {
                return false; // Executed by the dedicated planner after placing the Phantom.
            }
            let slots = self.board_size.saturating_sub(self.p.board.len()).max(1);
            if undead_apm::best_sacrifice(self.p, id(m), slots).is_none() {
                return false;
            }
        }
        !self.undead_specialist || undead_apm::can_play_minion(m, turn)
    }

    pub fn with_reborn_conversion(mut self, enabled: bool) -> Self {
        self.reborn_conversion = self.undead_finance && enabled;
        self
    }

    pub fn with_advanced(mut self, enabled: bool) -> Self {
        self.advanced = enabled;
        self
    }

    fn converting_reborn(&self) -> bool {
        self.reborn_conversion && undead_apm::phantom_growth_ready(self.p)
    }

    fn sacrifice_purchase(&self, card: &str) -> i32 {
        if self.reborn_conversion {
            undead_apm::reborn_purchase(self.p, card, self.board_size)
        } else {
            undead_apm::sacrifice_purchase(self.p, card, self.board_size)
        }
    }

    pub fn play_position(&self, m: &Minion) -> usize {
        if self.undead_finance && matches!(id(m), GRAVEROBBER | MAW_CASTER | BELLRINGER) {
            0
        } else {
            self.p.board.len()
        }
    }

    pub fn is_cycling(&self) -> bool {
        self.ready || has(&self.p.board, ESCAPEE)
    }

    fn financing(&self) -> bool {
        self.undead_finance && self.ready && has(&self.p.board, WEAVER)
    }

    pub fn action_budget(&self, game: &Game, seat: usize) -> u32 {
        match game.action_cap_of(seat) {
            u32::MAX => 141,
            cap => cap,
        }
    }

    fn wants_seven(&self) -> bool {
        match self.route {
            Route::Mech => presence(self.p, BEATBOXER) < 3,
            Route::Deity => presence(self.p, FEAR) < 3,
            _ => false,
        }
    }

    fn best_copy(&self, m: &Minion) -> bool {
        !self.p.board.iter().any(|other| {
            id(other) == id(m) && (other.golden, other.hp, other.eid) > (m.golden, m.hp, m.eid)
        })
    }

    fn component(&self, m: &Minion) -> f64 {
        let card = id(m);
        let value = match card {
            ROGERS => 35.0,
            PRIVATEER => {
                if self.rogers {
                    30.0
                } else {
                    17.0
                }
            }
            ESCAPEE => 25.0,
            BRANN => 12.0,
            CONNIVER if self.wants_seven() => {
                if m.golden {
                    5.0
                } else {
                    9.0
                }
            }
            _ => self
                .route
                .pieces()
                .iter()
                .find(|(c, _)| *c == card)
                .map_or(0.0, |(_, v)| {
                    f64::from(*v) * if self.is_cycling() { 1.3 } else { 0.85 }
                }),
        };
        value
            * if self.best_copy(m) || card == ROGERS {
                1.0
            } else {
                0.20
            }
    }

    fn body(&self, m: &Minion) -> f64 {
        (f64::from(m.atk.max(0)) + f64::from(m.hp.max(0))).sqrt()
            + if m.has(Kw::DIVINE_SHIELD) { 8.0 } else { 0.0 }
            + if m.has(Kw::REBORN) { 3.0 } else { 0.0 }
            + if m.has(Kw::VENOMOUS) { 9.0 } else { 0.0 }
            + self.keep_bonus(m) * 0.6
    }

    fn donors(&self) -> usize {
        self.p
            .shop
            .iter()
            .filter(|s| s.item.as_minion().is_some())
            .count()
    }

    fn cycle_value(&self, m: &Minion) -> i32 {
        if self.undead_finance {
            let card = id(m);
            let fodder = if self.converting_reborn() {
                undead_apm::reborn_fodder_value(self.p, m)
                    .max(undead_apm::extra_phantom_value(self.p, m))
            } else {
                undead_apm::fodder_value(self.p, m)
            };
            let special = self.sacrifice_purchase(card).max(fodder);
            let finance = if !self.financing() {
                0
            } else if card == BRANN
                && !has(&self.p.board, BRANN)
                && self.p.stats.undead_atk < 100
                && self.p.health + self.p.armor >= 18
            {
                24
            } else {
                undead_apm::finance_minion(self.p, m)
            };
            if special > 0 || finance > 0 {
                return special.max(finance);
            }
        }
        if matches!(id(m), "BG26_525" | "BG28_303" | "BG32_340") {
            return 0;
        }
        let brann = has(&self.p.board, BRANN);
        let worth = match id(m) {
            _ if self.advanced
                && self.route != Route::Undead
                && self.ready
                && (self.route.payoff(&self.p.board) || self.route == Route::Golden)
                && undead_apm::finance_minion(self.p, m) >= 20 =>
            {
                undead_apm::finance_minion(self.p, m)
            }
            "BG33_821" if self.undead_specialist && has(&self.p.board, WEAVER) => {
                if brann {
                    37
                } else {
                    25
                }
            }
            "BG36_343" | "BG33_821" | "BG28_550" => {
                if brann {
                    24
                } else {
                    15
                }
            }
            "BG36_520" => {
                if brann {
                    21
                } else {
                    12
                }
            }
            "BG22_202" | "BGS_115" | "BG36_100" | "BG36_308" | "BG24_715" => 15,
            "BG34_170" if self.route == Route::Mech => 22,
            "BG33_830" | "BG32_860" if self.route == Route::Tavern => 17,
            "BG34_682" if self.route == Route::Tavern => 23,
            CONNIVER if self.wants_seven() => {
                if presence(self.p, CONNIVER) >= 3 {
                    27
                } else {
                    13
                }
            }
            NAVIGATOR if self.wants_seven() && presence(self.p, CONNIVER) >= 3 => 32,
            _ if m.triple_reward && m.golden => 25,
            _ => 0,
        };
        worth + if m.golden && worth > 0 { 6 } else { 0 }
    }

    pub fn priority_piece(&self, m: &Minion) -> bool {
        let card = id(m);
        (matches!(card, ROGERS | PRIVATEER | ESCAPEE)
            || self.component(m) >= 9.0 && self.route.pieces().iter().any(|(c, _)| *c == card))
            && !has(&self.p.board, card)
    }

    pub fn magnetic_target(
        &self,
        hand: usize,
        m: &Minion,
        legal: &dyn Fn(Action) -> bool,
    ) -> Option<Action> {
        if !m.has(Kw::MAGNETIC) {
            return None;
        }
        let score = |target: &Minion| {
            let card = id(target);
            let preference = match self.route {
                Route::Mech => {
                    let host = match card {
                        BEATBOXER => -250.0,
                        DRONE => 52.0,
                        REVEILLE => 31.0,
                        _ => 0.0,
                    };
                    host + if target.has(Kw::DIVINE_SHIELD) {
                        17.0
                    } else {
                        0.0
                    }
                }
                Route::Undead if card == HANDLESS && !target.has(Kw::REBORN) => 120.0,
                _ => 0.0,
            };
            if self.undead_finance && id(m) == PROSTHETIC_HAND {
                let mut preference = undead_apm::prosthetic_target(self.p, target);
                if self.reborn_conversion {
                    preference =
                        preference.max(undead_apm::reborn_magnetic_priority(self.p, target));
                }
                return if preference > 0 {
                    f64::from(preference)
                } else {
                    -1.0
                };
            }
            preference + self.body(target).min(35.0)
        };
        self.p
            .board
            .iter()
            .enumerate()
            .filter(|(board, _)| {
                legal(Action::Magnetize {
                    hand,
                    board: *board,
                })
            })
            .max_by(|(_, a), (_, b)| score(a).total_cmp(&score(b)))
            .map(|(board, _)| Action::Magnetize { hand, board })
    }

    pub fn recycle(
        &self,
        game: &Game,
        seat: usize,
        legal: &dyn Fn(Action) -> bool,
    ) -> Option<Action> {
        let p = self.p;
        let roomy = p.actions_this_turn + 7 < self.action_budget(game, seat);
        if self.wants_seven()
            && p.hand.len() + 1 < game.cfg.hand_size
            && let Some((i, _)) = p
                .board
                .iter()
                .enumerate()
                .find(|(i, m)| id(m) == CONNIVER && m.golden && legal(Action::Sell(*i)))
        {
            return Some(Action::Sell(i));
        }
        if !self.is_cycling() || !roomy || p.tier < 5 {
            return None;
        }
        let weakest = p
            .board
            .iter()
            .enumerate()
            .filter(|(i, m)| !self.protects(m) && legal(Action::Sell(*i)))
            .min_by(|(_, a), (_, b)| self.body(a).total_cmp(&self.body(b)));
        let weak_body = weakest.map_or(f64::INFINITY, |(_, m)| self.body(m));
        let have_cycle = p.hand.iter().any(|h| {
            h.locked == 0
                && h.item
                    .as_minion()
                    .is_some_and(|m| self.cycle_value(m) >= 12)
        });
        let waiting_for_slot = p
            .hand
            .iter()
            .any(|h| h.locked == 0 && h.item.as_minion().is_some_and(|m| self.priority_piece(m)));
        if p.board.len() >= game.cfg.board_size
            && let Some((index, _)) = weakest
            && (have_cycle && weak_body <= 23.0
                || waiting_for_slot && (weak_body <= 45.0 || p.health + p.armor >= 22))
        {
            return Some(Action::Sell(index));
        }
        let mut best = None;
        let mut best_value = 0;
        for (i, slot) in p.shop.iter().enumerate() {
            let action = Action::Buy(i);
            let Some((price, health, _)) = game.slot_price(seat, i) else {
                continue;
            };
            if !legal(action) || health && p.health + p.armor < 20 {
                continue;
            }
            let value = match &slot.item {
                Item::Spell { card, .. } if self.undead_finance && normal(*card) == BUTCHERING => {
                    self.sacrifice_purchase(BUTCHERING)
                }
                Item::Spell { card, .. }
                    if self.financing() && undead_apm::finance_spell(p, *card) > 0 =>
                {
                    undead_apm::finance_spell(p, *card)
                }
                Item::Spell { card, .. } => match normal(*card) {
                    TOMB_TURNING
                        if self.undead_specialist
                            && undead_apm::needs_undead_discover(p)
                            && p.actions_this_turn + 10 < self.action_budget(game, seat) =>
                    {
                        36
                    }
                    "BG28_810" | "BG36_303" | "BG28_571" => 18,
                    "BG28_827" | "BG28_GIL_836" => 14,
                    "BG28_849"
                        if self.undead_specialist
                            && has(&p.board, WEAVER)
                            && undead_apm::finishers_collected(p)
                            && (!self.reborn_conversion || undead_apm::reborn_shop_ready(p))
                            && p.gold >= price + 2
                            && p.actions_this_turn + 12 < self.action_budget(game, seat) =>
                    {
                        25
                    }
                    "BG28_849" if self.undead_specialist => 0,
                    "BG28_849" if self.route != Route::Mech && !self.undead_specialist => 12,
                    "EBG_Spell_017" if self.spell_bonus(*card) >= 18.0 => 36,
                    _ if price <= 1 || self.route == Route::Undead && price <= 2 => 8,
                    _ => 0,
                },
                Item::Minion(m) if self.priority_piece(m) => 42,
                Item::Minion(m) if self.undead_finance && id(m) == PROSTHETIC_HAND => {
                    self.cycle_value(m)
                }
                Item::Minion(m)
                    if self.converting_reborn()
                        && weakest.is_some()
                        && (undead_apm::is_sacrifice(id(m))
                            || undead_apm::reborn_fodder_value(p, m) > 0
                            || undead_apm::extra_phantom_value(p, m) > 0) =>
                {
                    self.cycle_value(m)
                }
                Item::Minion(m) if p.board.len() < game.cfg.board_size || weak_body <= 23.0 => {
                    self.cycle_value(m)
                }
                _ => 0,
            } - 2 * price;
            if value > best_value {
                best = Some(action);
                best_value = value;
            }
        }
        if best.is_some() {
            return best;
        }
        None
    }

    pub fn prepare(
        &self,
        game: &Game,
        seat: usize,
        legal: &dyn Fn(Action) -> bool,
    ) -> Option<Action> {
        if self.route == Route::Mech {
            for (board, m) in self.p.board.iter().enumerate() {
                if id(m) == DRONE
                    && m.counter == 0
                    && legal(Action::Activate(board))
                    && self
                        .p
                        .hand
                        .iter()
                        .enumerate()
                        .any(|(hand, _)| legal(Action::Magnetize { hand, board }))
                {
                    return Some(Action::Activate(board));
                }
            }
        }
        if self.ready
            && self.route == Route::Tavern
            && self.route.payoff(&self.p.board)
            && self.donors() == 0
            && legal(Action::Refresh)
            && self.refresh(game, seat)
            && self.p.hand.iter().enumerate().any(|(hand, h)| {
                matches!(h.item, Item::Spell { .. }) && legal(Action::Play { hand, pos: 0 })
            })
        {
            return Some(Action::Refresh);
        }
        None
    }

    pub fn refresh(&self, game: &Game, seat: usize) -> bool {
        self.is_cycling()
            && self.p.actions_this_turn + 5 < self.action_budget(game, seat)
            && game
                .malchezaar(seat)
                .is_none_or(|_| self.p.health + self.p.armor > 24)
            && (self.p.tier >= 5 && self.p.gold > game.refresh_price(seat)
                || self.rogers
                    && self
                        .p
                        .board
                        .iter()
                        .any(|m| id(m) == ROGERS && m.counter >= 8)
                || self.route == Route::Tavern && self.p.free_refreshes > 0 && self.donors() < 3)
    }
}

impl BuildPlan for Plan<'_> {
    fn protects(&self, m: &Minion) -> bool {
        let card = id(m);
        if self.undead_finance
            && (card == PHANTOM && m.max_hp >= 60
                || card == BRANN
                    && self.best_copy(m)
                    && self.p.stats.undead_atk < 100
                    && self.ready)
        {
            return true;
        }
        if self.converting_reborn()
            && (undead_apm::reborn_fodder_value(self.p, m) > 0
                || undead_apm::extra_phantom_value(self.p, m) > 0)
        {
            return true;
        }
        self.best_copy(m)
            && (matches!(card, ROGERS | PRIVATEER)
                || card == ESCAPEE && !self.ready
                || self.route.critical(card)
                || self.route == Route::Deity
                    && self
                        .p
                        .board
                        .iter()
                        .filter(|m| m.is(Races::ABERRATION))
                        .count()
                        <= 4
                    && m.is(Races::ABERRATION)
                    && self.route.payoff(&self.p.board))
    }

    fn keep_bonus(&self, m: &Minion) -> f64 {
        let specialized = if self.undead_specialist {
            undead_apm::keep_bonus(self.p, m)
        } else {
            0.0
        };
        self.component(m)
            + specialized
            + match self.route {
                Route::Undead if m.is(Races::UNDEAD) => {
                    3.0 + if m.has(Kw::REBORN) { 5.0 } else { 0.0 }
                }
                Route::Tavern
                    if m.is_card_id(db().id("BG35_155"))
                        || self.donors() > 0 && id(m) == "BG36_621" =>
                {
                    9.0
                }
                Route::Mech if m.is(Races::MECH) && m.has(Kw::DIVINE_SHIELD) => 7.0,
                Route::Deity if m.is(Races::ABERRATION) => {
                    13.0 + if m.has(Kw::REBORN) { 8.0 } else { 0.0 }
                }
                _ => 0.0,
            }
    }

    fn acquisition_bonus(&self, m: &Minion, copies: usize) -> f64 {
        let card = id(m);
        let combo = match card {
            HANDLESS | WEAVER | ROGERS | PRIVATEER if self.undead_specialist && copies == 2 => 22.0,
            BELLRINGER if self.undead_specialist && has(&self.p.board, PHANTOM) => 11.0,
            CONNIVER if self.wants_seven() => match copies {
                2 => 38.0,
                1 => 13.0,
                _ => 6.0,
            },
            NAVIGATOR if self.wants_seven() && presence(self.p, CONNIVER) >= 3 => 33.0,
            "BG36_342" if self.is_cycling() && self.route == Route::Undead => 13.0,
            _ if self.priority_piece(m) => 12.0,
            _ if matches!(card, PRIVATEER | BRANN) && has(&self.p.board, card) && copies != 2 => {
                -12.0
            }
            _ => 0.0,
        };
        combo
            + if self.is_cycling() {
                f64::from(self.cycle_value(m)) * 0.45
            } else {
                0.0
            }
    }

    fn spell_bonus(&self, card: CardId) -> f64 {
        if self.undead_finance && normal(card) == BUTCHERING {
            let bonus = self.sacrifice_purchase(BUTCHERING);
            return if bonus > 0 {
                f64::from(bonus) * 0.65
            } else {
                -40.0
            };
        }
        let ordinary = match normal(card) {
            "BG31_896" if self.wants_seven() => 34.0,
            "EBG_Spell_017" if self.wants_seven() && has(&self.p.board, CONNIVER) => 30.0,
            "EBG_Spell_017"
                if self.undead_specialist && undead_apm::wants_golden_handless(self.p) =>
            {
                28.0
            }
            "EBG_Spell_017"
                if self.route == Route::Undead
                    && !self.undead_specialist
                    && has(&self.p.board, HANDLESS) =>
            {
                18.0
            }
            "BG28_810" | "BG28_827" | "BG36_303" if self.is_cycling() => 8.0,
            "BG34_689" | "BG34_444" | "BG28_607" if self.route == Route::Tavern => 16.0,
            _ if self.is_cycling() && is_tavern_spell(card) => 3.0,
            _ => 0.0,
        };
        ordinary
            + if self.advanced
                && self.route != Route::Undead
                && self.ready
                && self.route.payoff(&self.p.board)
            {
                f64::from(undead_apm::finance_spell(self.p, card)) * 0.25
            } else {
                0.0
            }
            + if self.undead_specialist {
                undead_apm::growth_spell_bonus(self.p, card)
            } else {
                0.0
            }
            + if self.financing() {
                f64::from(undead_apm::finance_spell(self.p, card)) * 0.55
            } else {
                0.0
            }
    }

    fn hand_priority(&self, item: &Item) -> f64 {
        match item {
            Item::Minion(m)
                if self.undead_specialist && id(m) == WEAVER && !has(&self.p.board, WEAVER) =>
            {
                180.0
            }
            Item::Minion(m)
                if self.priority_piece(m) || id(m) == BRANN && !has(&self.p.board, BRANN) =>
            {
                125.0
            }
            Item::Minion(m) if self.cycle_value(m) >= 12 && self.is_cycling() => 12.0,
            Item::Spell { card, .. } if is_bounty(*card) => {
                if normal(*card) == "BG33_815" || self.p.hand.len() >= 7 {
                    90.0
                } else {
                    25.0
                }
            }
            _ => 0.0,
        }
    }

    fn can_cast(&self, card: CardId) -> bool {
        if self.reborn_conversion
            && normal(card) == "BG28_849"
            && !undead_apm::reborn_shop_ready(self.p)
        {
            return false;
        }
        if self.undead_finance && normal(card) == BUTCHERING {
            if self.converting_reborn() {
                return false;
            }
            let slots = self.board_size.saturating_sub(self.p.board.len()) + 1;
            return undead_apm::best_sacrifice(self.p, BUTCHERING, slots).is_some();
        }
        !self.undead_specialist
            || normal(card) != "BG28_849"
            || undead_apm::finishers_collected(self.p)
            || self.p.hand.len() >= 8 && self.p.shop.iter().all(|s| s.item.as_minion().is_none())
    }

    fn activate(&self, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        if self.p.hand.len() >= 7 {
            return None;
        }
        self.p.board.iter().enumerate().find_map(|(i, m)| {
            let action = Action::Activate(i);
            let useful = match (self.route, id(m)) {
                (Route::Tavern, GEOMANT) => has(&self.p.board, FELBOAR) && self.p.hand.len() <= 4,
                (Route::Tavern, "BG36_621") => self.donors() >= 3,
                (Route::Mech, DRONE) => self
                    .p
                    .hand
                    .iter()
                    .any(|h| h.item.as_minion().is_some_and(|m| m.has(Kw::MAGNETIC))),
                (_, "BG36_342") => self.is_cycling(),
                _ => false,
            };
            (useful && legal(action)).then_some(action)
        })
    }

    fn pending(&self, menu: &Pending, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        if self.undead_finance
            && let Pending::ChooseOne { card, .. } = menu
            && normal(*card) == "BG31_890"
        {
            let choice = usize::from(undead_apm::finishers_collected(self.p) && self.p.tier == 5);
            return legal(Action::Choose(choice)).then_some(Action::Choose(choice));
        }
        let Pending::Target { options, cont } = menu else {
            return None;
        };
        let card = match cont {
            Continuation::HandSpell { hid, .. } => {
                normal(self.p.hand.iter().find(|h| h.hid == *hid)?.item.card())
            }
            Continuation::Battlecry { eid, .. } => {
                id(self.p.board.iter().find(|m| m.eid == *eid)?)
            }
            Continuation::Activate { eid } if self.undead_specialist => {
                id(self.p.board.iter().find(|m| m.eid == *eid)?)
            }
            _ => return None,
        };
        if self.undead_finance && undead_apm::is_sacrifice(card) {
            let slots = self.board_size.saturating_sub(self.p.board.len()) + 1;
            return options
                .iter()
                .enumerate()
                .filter_map(|(i, eid)| {
                    let action = Action::Choose(i);
                    let m = self.p.board.iter().find(|m| m.eid == *eid)?;
                    let score = if self.reborn_conversion {
                        undead_apm::reborn_target_score(self.p, card, m, slots)
                    } else {
                        undead_apm::sacrifice_score(self.p, card, m, slots)
                    }?;
                    legal(action).then_some((action, score))
                })
                .max_by_key(|(_, score)| *score)
                .map(|(action, _)| action)
                .or_else(|| legal(Action::Cancel).then_some(Action::Cancel));
        }
        if self.undead_specialist && card == BELLRINGER {
            return options
                .iter()
                .enumerate()
                .filter_map(|(i, eid)| {
                    let action = Action::Choose(i);
                    let m = self.p.board.iter().find(|m| m.eid == *eid)?;
                    (legal(action) && undead_apm::bellringer_target(self.p, m))
                        .then_some((action, db().get(m.card).atk - m.hp))
                })
                .max_by_key(|(_, value)| *value)
                .map(|(action, _)| action);
        }
        let score = |m: &Minion| -> f64 {
            match card {
                NAVIGATOR | "EBG_Spell_017" if id(m) == CONNIVER && self.wants_seven() => 200.0,
                "EBG_Spell_017" if self.route == Route::Undead && id(m) == HANDLESS => {
                    160.0
                        + if self.undead_specialist && m.has(Kw::REBORN) {
                            65.0
                        } else {
                            0.0
                        }
                }
                "BG20_GEM" | "BG28_607" if self.route == Route::Tavern && id(m) == DISTRACTOR => {
                    130.0
                }
                "BG20_GEM" | "BG28_607" | "BG23_357"
                    if self.route == Route::Tavern && id(m) == FELBOAR =>
                {
                    100.0
                }
                "BG26_814" if id(m) == ROGERS => 80.0,
                _ => 0.0,
            }
        };
        options
            .iter()
            .enumerate()
            .filter(|(i, _)| legal(Action::Choose(*i)))
            .filter_map(|(i, id)| {
                self.p
                    .board
                    .iter()
                    .find(|m| m.eid == *id)
                    .map(|m| (i, score(m)))
            })
            .filter(|(_, value)| *value > 0.0)
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| Action::Choose(i))
    }

    fn upgrade(&self, _game: &Game, _seat: usize, ordinary: bool) -> bool {
        ordinary
    }
    fn search_missing(&self, _game: &Game, _seat: usize) -> bool {
        false
    }

    fn attack_bonus(&self, m: &Minion) -> f64 {
        match (self.route, id(m)) {
            (Route::Undead, HANDLESS) => 24.0,
            (Route::Undead, PHANTOM) => -180.0,
            (Route::Undead, WEAVER | ROGERS | PRIVATEER) if self.undead_specialist => -7.0,
            (Route::Deity, _) if m.is(Races::ABERRATION) && id(m) != FEAR => 7.0,
            _ => 0.0,
        }
    }
}

pub fn tier_seven_choice(
    game: &Game,
    seat: usize,
    menu: &Pending,
    route: Option<Route>,
    legal: &dyn Fn(Action) -> bool,
) -> Option<Action> {
    let targets = [(Route::Mech, BEATBOXER), (Route::Deity, FEAR)];
    if !targets
        .iter()
        .any(|(r, _)| game.lobby_races.contains(r.required()))
    {
        return None;
    }
    match menu {
        Pending::HeroSelect { options } => options
            .iter()
            .enumerate()
            .find(|(i, card)| normal(**card) == "BG27_HERO_801" && legal(Action::Choose(*i)))
            .map(|(i, _)| Action::Choose(i)),
        Pending::Discover { options, .. } => options
            .iter()
            .enumerate()
            .filter(|(i, _)| legal(Action::Choose(*i)))
            .filter_map(|(i, item)| {
                let m = item.as_minion()?;
                let (r, _) = targets
                    .iter()
                    .find(|(r, card)| game.lobby_races.contains(r.required()) && id(m) == *card)?;
                let p = &game.players[seat];
                let score = 20 * i32::from(route == Some(*r)) + presence(p, r.pieces()[0].0);
                Some((i, score))
            })
            .max_by_key(|(_, score)| *score)
            .map(|(i, _)| Action::Choose(i)),
        _ => None,
    }
}
