use crate::cards::{CardId, db};
use crate::content::is_tavern_spell;
use crate::script::{TargetSpec, script};
use crate::state::{Continuation, Item, Minion, Pending, Player};
use crate::types::{Kw, Races};
use crate::{Action, Game};

use super::catalog::{Route::*, *};
use super::guide_bot::Plan as BuildPlan;

pub struct Plan<'a> {
    p: &'a Player,
    pub route: Route,
    turn: u32,
    ready: bool,
    advanced: bool,
}

impl<'a> Plan<'a> {
    pub fn new(p: &'a Player, route: Route, turn: u32) -> Self {
        Self {
            p,
            route,
            turn,
            ready: route.core(p, &p.board),
            advanced: false,
        }
    }

    pub fn with_advanced(mut self, advanced: bool) -> Self {
        self.advanced = advanced;
        self
    }

    fn active_supply(&self, m: &Minion) -> i32 {
        if !self.advanced || !self.ready || self.spare_hand() < 2 {
            return 0;
        }
        let card = key(m.card);
        let brann = self.has(BRANN);
        let battlecry = script(m.card).is_some_and(|s| s.battlecry.is_some());
        match self.route {
            Keywords | Tidecaller
                if matches!(card, "BG32_336" | "BGS_020") && self.count(Races::MURLOC) >= 2 =>
            {
                11 + 6 * i32::from(brann)
            }
            Keywords if matches!(card, "BG31_148" | KELP | SALVAGER) => 8,
            Magicfin if card == APPRENTICE && !self.has(EYE) => 14,
            Lobster | Beetle | BeastSummon | DeityDeath
                if card == SHARK && self.has(GRIFFIN) && self.has(BIRD) =>
            {
                19
            }
            Behemoth | Beetle | BeastSummon
                if m.is(Races::BEAST)
                    && (m.has(Kw::REBORN) || (self.count(Races::BEAST) >= 3 && card == RAT)) =>
            {
                7
            }
            UndeadAttack | Knight
                if m.is(Races::UNDEAD) && m.has(Kw::REBORN) && self.p.stats.undead_atk >= 15 =>
            {
                7
            }
            DemonShop | DemonDeath
                if matches!(card, "BG36_503" | "BG36_621" | "BG35_155") && self.donors() >= 2 =>
            {
                9
            }
            Tempest | Ultraviolet
                if m.is(Races::ELEMENTAL)
                    && matches!(card, CYCLONE | "BGS_115" | "BG31_815" | "BG36_181") =>
            {
                12 + 5 * i32::from(brann && battlecry)
            }
            MechSpell | Volumizer if m.has(Kw::MAGNETIC) && self.count(Races::MECH) > 0 => 10,
            Shield | Hope if m.has(Kw::DIVINE_SHIELD) && m.atk >= 6 => 7,
            Poison if tech(m) && self.pressure() => 12,
            Barrage | HandGems
                if matches!(card, REFINDER | SURVEYOR | GEOMANT)
                    && self.p.stats.gem_atk + self.p.stats.gem_hp >= 5 =>
            {
                9
            }
            Sludge
                if matches!(card, "BG36_300" | "BG36_312")
                    && self.p.hand.iter().any(|h| {
                        matches!(key(h.item.card()), "BG36_301t" | "BG36_371" | "BG36_303")
                    }) =>
            {
                9
            }
            DragonSpell
                if matches!(card, "BG36_245" | "BG32_821" | KELP)
                    && self.count(Races::DRAGON) >= 3 =>
            {
                8
            }
            DragonCry
                if battlecry
                    && self.count(Races::DRAGON) >= 3
                    && (brann || m.is(Races::DRAGON)) =>
            {
                10 + 5 * i32::from(m.is(Races::DRAGON))
            }
            _ => 0,
        }
    }

    fn cycle_supply(&self, m: &Minion) -> bool {
        if !self.advanced {
            return match self.route {
                Tempest | Ultraviolet => {
                    m.is(Races::ELEMENTAL)
                        && matches!(key(m.card), CYCLONE | "BGS_115" | "BG31_815" | "BG36_181")
                }
                DragonCry => script(m.card).is_some_and(|s| s.battlecry.is_some()),
                Keywords => matches!(key(m.card), "BG31_148" | "BG32_336" | "BGS_020"),
                Magicfin => is(m, APPRENTICE) && !self.has(EYE),
                _ => is(m, SHARK) && self.has(GRIFFIN) && self.has(BIRD),
            };
        }
        self.active_supply(m) >= 11 && !self.priority_piece(m)
    }

    fn finish_core(
        &self,
        game: &Game,
        seat: usize,
        legal: &dyn Fn(Action) -> bool,
    ) -> Option<Action> {
        if !self.advanced
            || !self.ready
            || self.pressure()
            || self.p.tier < 4
            || self.p.actions_this_turn + 5 >= game.action_cap_of(seat)
            || !self.route.full_pool(game.lobby_races)
        {
            return None;
        }
        let spec = self.route.spec();
        let complete = |m: &Minion| {
            self.priority_piece(m) && (spec.full.contains(&id(m)) || id(m) == spec.anchor)
        };
        let full_board = self.p.board.len() == game.cfg.board_size;
        let weakest = self
            .p
            .board
            .iter()
            .enumerate()
            .filter(|(i, m)| legal(Action::Sell(*i)) && !self.protects(m))
            .min_by(|(_, a), (_, b)| Self::stat(a).total_cmp(&Self::stat(b)));
        if full_board && weakest.is_none_or(|(_, m)| Self::stat(m) > 16.0) {
            return None;
        }
        if self
            .p
            .hand
            .iter()
            .any(|h| h.locked == 0 && h.item.as_minion().is_some_and(complete))
        {
            return full_board
                .then(|| weakest.map(|(i, _)| Action::Sell(i)))
                .flatten();
        }
        self.p
            .shop
            .iter()
            .enumerate()
            .filter(|(i, s)| {
                legal(Action::Buy(*i))
                    && s.item.as_minion().is_some_and(complete)
                    && game
                        .slot_price(seat, *i)
                        .is_some_and(|(gold, health, _)| gold <= 3 && !health)
            })
            .max_by_key(|(_, s)| self.card_value(db().normal_of(s.item.card())) as i32)
            .map(|(i, _)| Action::Buy(i))
    }

    fn has(&self, name: &str) -> bool {
        has_key(&self.p.board, name)
    }
    fn pressure(&self) -> bool {
        self.p.health + self.p.armor <= 15
            || self.p.health + self.p.armor <= 23 && self.p.stats.last_damage <= -9
    }
    fn count(&self, r: Races) -> usize {
        self.p.board.iter().filter(|m| m.is(r)).count()
    }
    fn donors(&self) -> usize {
        self.p
            .shop
            .iter()
            .filter(|s| s.item.as_minion().is_some())
            .count()
    }
    fn spare_hand(&self) -> usize {
        10_usize.saturating_sub(self.p.hand.len())
    }
    fn rally_route(&self) -> bool {
        matches!(self.route, Lobster | BeastSummon | DeityDeath)
    }
    fn target_left(&self) -> Option<&str> {
        match self.route {
            Lobster => Some("BG36_202"),
            BeastSummon => Some("BGS_018"),
            DeityDeath => Some("BG36_318"),
            _ => None,
        }
    }
    fn card_value(&self, card: CardId) -> f64 {
        let s = self.route.spec();
        if card == s.anchor {
            23.0
        } else if s.essential(card) {
            16.0
        } else if s.core.iter().any(|g| g.contains(&card)) {
            11.0
        } else if s.support.contains(&card) {
            6.0
        } else {
            0.0
        }
    }
    fn stat(m: &Minion) -> f64 {
        (f64::from(m.atk.max(0)) + f64::from(m.hp.max(0))).sqrt()
            + 6.0 * f64::from(m.has(Kw::DIVINE_SHIELD))
            + 4.0 * f64::from(m.has(Kw::WINDFURY) || m.has(Kw::CLEAVE))
    }
    fn best_copy(&self, m: &Minion) -> bool {
        !self.p.board.iter().any(|b| {
            id(b) == id(m) && (b.golden, b.atk + b.hp, b.eid) > (m.golden, m.atk + m.hp, m.eid)
        })
    }
    fn has_rewind(&self) -> bool {
        self.has("BG26_174") || self.has("BG32_873")
    }
    fn spell_host(&self, m: &Minion) -> f64 {
        let typed = |r| if m.is(r) { 40.0 } else { 0.0 };
        match self.route {
            DemonShop if is(m, DISTRACTOR) => 135.0,
            DemonShop | DemonDeath => typed(Races::DEMON),
            MechSpell | Shield => typed(Races::MECH),
            Tidecaller | Keywords => typed(Races::MURLOC),
            HandGems if is(m, "BG28_583") => 120.0,
            Hope | DragonSpell | DragonCry => typed(Races::DRAGON),
            _ => 0.0,
        }
    }
    pub fn priority_piece(&self, m: &Minion) -> bool {
        let card = id(m);
        let targets = self.route.spec();
        targets.essential(card)
            && (!has(&self.p.board, card)
                || self.route == Shield
                    && is(m, CZAR)
                    && self.p.board.iter().filter(|b| is(b, CZAR)).count() == 1)
    }
    pub fn needs_six(&self, lobby: Races) -> bool {
        let s = self.route.spec();
        s.full.iter().chain(std::iter::once(&s.anchor)).any(|&c| {
            let d = db().get(c);
            d.tier == 6
                && !has(&self.p.board, c)
                && (d.races.is_none() || d.races.intersects(lobby))
                && (d.assoc.is_none() || d.assoc.intersects(lobby))
        })
    }
    pub fn can_play_minion(&self, m: &Minion) -> bool {
        if !self.has_rewind() && self.pressure() && is(m, "BG26_525") {
            return false;
        }
        !self.keep_in_hand(m)
    }

    fn keep_in_hand(&self, m: &Minion) -> bool {
        if self.route != Tidecaller
            || self.p.board.len() < 5
            || self.spare_hand() < 3
            || self.priority_piece(m)
            || !m.is(Races::MURLOC)
        {
            return false;
        }
        let wants_payload = self.has(HUNGRY) || self.has("BG26_354") || self.ready;
        wants_payload
            && self
                .p
                .hand
                .iter()
                .filter_map(|h| h.item.as_minion())
                .filter(|h| h.is(Races::MURLOC) && !self.priority_piece(h))
                .filter(|h| (Self::stat(h), h.eid) > (Self::stat(m), m.eid))
                .count()
                < 2
    }
    pub fn magnetic(
        &self,
        p: &Player,
        hand: usize,
        legal: &dyn Fn(Action) -> bool,
    ) -> Option<Action> {
        let m = p.hand.get(hand)?.item.as_minion()?;
        if !m.has(Kw::MAGNETIC) {
            return None;
        }
        let value = |t: &Minion| {
            self.spell_host(t)
                + Self::stat(t).min(50.0)
                + if is(t, "BG36_506") && t.counter > 0 {
                    90.0
                } else {
                    0.0
                }
                - if is(t, "BG26_149") { 200.0 } else { 0.0 }
        };
        p.board
            .iter()
            .enumerate()
            .filter(|(b, _)| legal(Action::Magnetize { hand, board: *b }))
            .max_by(|(_, a), (_, b)| value(a).total_cmp(&value(b)))
            .map(|(board, _)| Action::Magnetize { hand, board })
    }

    fn toss_value(&self, m: &Minion) -> f64 {
        let useful = if self.route == UndeadAttack || self.route == Knight {
            if is(m, "BG28_309") {
                18.0
            } else if m.has(Kw::REBORN) {
                10.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        useful - Self::stat(m) - self.card_value(id(m)) * 0.8
    }

    fn battlecry_target(&self, m: &Minion) -> f64 {
        let bonus = match self.route {
            Keywords if is(m, "BG31_148") => 30.0,
            Tempest | Ultraviolet if is(m, CYCLONE) => 24.0,
            Magicfin if is(m, APPRENTICE) => 28.0,
            DragonCry | DragonSpell if is(m, "BG34_633") => 26.0,
            Tidecaller if is(m, "BGS_020") => 19.0,
            _ if is(m, "BG32_336") || is(m, "BG34_170") => 8.0,
            _ => 0.0,
        };
        bonus + 3.0 * f64::from(m.golden)
    }
    fn rally_target(&self, m: &Minion) -> f64 {
        let bonus = match self.route {
            DragonSpell if is(m, CHRONO) => 35.0,
            DragonSpell if is(m, "BG36_241") => 29.0,
            HandGems | Barrage if is(m, REFINDER) => 32.0,
            Keywords | Poison if is(m, POISON) => 31.0,
            DeityDeath | Lobster | Beetle | BeastSummon if is(m, GRIFFIN) => 25.0,
            _ => 0.0,
        };
        bonus + 2.0 * f64::from(m.golden)
    }
    fn usable(&self, m: &Minion) -> bool {
        match key(m.card) {
            KELP => {
                self.p.board.iter().any(|t| self.battlecry_target(t) >= 8.0)
                    && self.spare_hand() > 0
            }
            "BG36_243" => self.p.board.iter().any(|t| self.rally_target(t) >= 10.0),
            SALVAGER => self.count(Races::MURLOC) >= 2,
            LIONFISH => {
                self.rally_route()
                    && self.has(BIRD)
                    && self
                        .p
                        .board
                        .iter()
                        .find(|m| m.is(Races::BEAST))
                        .is_some_and(|m| is(m, GRIFFIN))
                    && self.target_left().is_some_and(|id| {
                        self.p
                            .board
                            .iter()
                            .find(|m| script(m.card).is_some_and(|s| s.deathrattle.is_some()))
                            .is_some_and(|m| is(m, id))
                    })
            }
            "BG36_362" => matches!(self.route, DemonDeath | DemonShop) && self.turn <= 14,
            "BG36_503" => {
                matches!(self.route, DemonDeath | DemonShop)
                    && self.donors() >= 2
                    && self.count(Races::DEMON) >= 2
            }
            "BG36_621" => self.donors() >= 3 && (self.p.stats.tavern_hp >= 5 || self.has(FELBOAR)),
            GEOMANT => {
                self.spare_hand() >= 2
                    && matches!(
                        self.route,
                        HandGems | MechSpell | Barrage | DemonShop | Tidecaller
                    )
            }
            "BG36_300" | "BG36_312" | "BG28_582" => self.p.hand.iter().any(|h| {
                h.locked == 0
                    && !h.held
                    && match &h.item {
                        Item::Spell { card, .. } => {
                            matches!(key(*card), "BG36_301t" | "BG36_371" | "BG36_303")
                        }
                        Item::Minion(m) => !self.priority_piece(m) && Self::stat(m) <= 4.0,
                    }
            }),
            "BG36_511" => {
                self.p.board.len() >= 6
                    && self.p.board.iter().any(|t| {
                        t.eid != m.eid && t.is(Races::UNDEAD) && self.toss_value(t) > -10.0
                    })
            }
            _ => true,
        }
    }
    pub fn allow_activate(&self, m: &Minion) -> bool {
        self.usable(m)
    }

    fn desired_order(&self, end: bool) -> Vec<usize> {
        let b = &self.p.board;
        let mut order: Vec<_> = (0..b.len()).collect();
        if let Some(left) = self.target_left()
            && self.has(BIRD)
            && let Some(index) = b.iter().position(|m| is(m, left))
        {
            order.retain(|&i| i != index);
            order.insert(0, index);
            if let Some(i) = b.iter().position(|m| is(m, GRIFFIN)) {
                order.retain(|&j| j != i);
                order.insert(
                    usize::from(left != "BG36_202" && b[index].is(Races::BEAST) && end),
                    i,
                );
                if !end && left == "BGS_018" {
                    // Griffin must be the left-most Beast or Fishbait will not trigger Rally.
                    if script(b[i].card).is_some_and(|s| s.deathrattle.is_none()) {
                        return order;
                    }
                }
            }
            return order;
        }
        if !end {
            return order;
        }
        if let Some(eye) = b.iter().position(|m| is(m, EYE))
            && matches!(self.route, Magicfin | Keywords | Tidecaller)
        {
            let mut targets: Vec<_> = order
                .iter()
                .copied()
                .filter(|&i| i != eye && self.battlecry_target(&b[i]) >= 8.0)
                .collect();
            targets.sort_by(|&a, &b_idx| {
                self.battlecry_target(&b[b_idx])
                    .total_cmp(&self.battlecry_target(&b[a]))
                    .then_with(|| b[a].eid.cmp(&b[b_idx].eid))
            });
            if !targets.is_empty() {
                targets.truncate(2);
                order.retain(|&i| i != eye && !targets.contains(&i));
                order.push(targets[0]);
                order.push(eye);
                if targets.len() == 2 {
                    order.push(targets[1]);
                }
                return order;
            }
        }
        if matches!(self.route, Hope | DragonSpell | DragonCry)
            && let Some(poet) = b.iter().position(|m| is(m, POET))
        {
            let value = |m: &Minion| {
                Self::stat(m)
                    + if is(m, BROODMOTHER) || is(m, "BG36_241") {
                        35.0
                    } else {
                        0.0
                    }
            };
            let mut dragons: Vec<_> = order
                .iter()
                .copied()
                .filter(|&i| i != poet && b[i].is(Races::DRAGON))
                .collect();
            dragons.sort_by(|&a, &b_idx| {
                value(&b[b_idx])
                    .total_cmp(&value(&b[a]))
                    .then_with(|| b[a].eid.cmp(&b[b_idx].eid))
            });
            if !dragons.is_empty() {
                dragons.truncate(2);
                order.retain(|&i| i != poet && !dragons.contains(&i));
                order.insert(0, dragons[0]);
                order.insert(1, poet);
                if dragons.len() == 2 {
                    order.insert(2, dragons[1]);
                }
                return order;
            }
        }
        order.sort_by(|&a, &b_idx| {
            let priority = |m: &Minion| Self::stat(m) * 0.5 + self.attack_bonus(m);
            priority(&b[b_idx])
                .total_cmp(&priority(&b[a]))
                .then_with(|| b[a].eid.cmp(&b[b_idx].eid))
        });
        order
    }

    pub fn position(&self, end: bool, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        self.desired_order(end)
            .into_iter()
            .enumerate()
            .find_map(|(to, from)| {
                let action = Action::Move { from, to };
                (to != from && legal(action)).then_some(action)
            })
    }

    pub fn prepare(
        &self,
        game: &Game,
        seat: usize,
        legal: &dyn Fn(Action) -> bool,
    ) -> Option<Action> {
        if self.rally_route()
            && self.has(BIRD)
            && self.has(GRIFFIN)
            && self.target_left().is_some_and(|m| self.has(m))
            && self
                .p
                .board
                .iter()
                .enumerate()
                .any(|(i, m)| is(m, LIONFISH) && legal(Action::Activate(i)))
        {
            if let Some(a) = self.position(false, legal) {
                return Some(a);
            }
            if let Some((i, _)) = self
                .p
                .board
                .iter()
                .enumerate()
                .find(|(i, m)| is(m, LIONFISH) && self.usable(m) && legal(Action::Activate(*i)))
            {
                return Some(Action::Activate(i));
            }
        }
        if matches!(self.route, MechSpell | Volumizer | Shield) {
            for (i, m) in self.p.board.iter().enumerate() {
                if is(m, "BG36_506")
                    && m.counter == 0
                    && legal(Action::Activate(i))
                    && self
                        .p
                        .hand
                        .iter()
                        .enumerate()
                        .any(|(hand, _)| legal(Action::Magnetize { hand, board: i }))
                {
                    return Some(Action::Activate(i));
                }
            }
        }
        if self.ready
            && matches!(self.route, DemonShop | DemonDeath | Barrage)
            && self.has(FELBOAR)
            && self.donors() == 0
            && self.p.gold > game.refresh_price(seat)
            && legal(Action::Refresh)
            && game
                .malchezaar(seat)
                .is_none_or(|_| self.has_rewind() || !self.pressure())
            && self.p.hand.iter().enumerate().any(|(hand, h)| {
                matches!(h.item, Item::Spell { .. }) && legal(Action::Play { hand, pos: 0 })
            })
        {
            return Some(Action::Refresh);
        }
        None
    }

    pub fn cycle(
        &self,
        game: &Game,
        seat: usize,
        legal: &dyn Fn(Action) -> bool,
    ) -> Option<Action> {
        if let Some(action) = self.finish_core(game, seat, legal) {
            return Some(action);
        }
        if !self.advanced
            && !matches!(
                self.route,
                Tempest
                    | Ultraviolet
                    | DragonCry
                    | Keywords
                    | Magicfin
                    | Lobster
                    | BeastSummon
                    | DeityDeath
            )
            || !self.ready
            || self.p.actions_this_turn >= if self.advanced { 58 } else { 43 }
        {
            return None;
        }
        let full = self.p.board.len() == game.cfg.board_size;
        let supply = |m: &Minion| self.cycle_supply(m);
        if full
            && self
                .p
                .hand
                .iter()
                .filter_map(|h| h.item.as_minion())
                .any(supply)
        {
            return self
                .p
                .board
                .iter()
                .enumerate()
                .filter(|(i, m)| legal(Action::Sell(*i)) && !self.protects(m))
                .filter(|(_, m)| Self::stat(m) < 13.0)
                .min_by(|(_, a), (_, b)| Self::stat(a).total_cmp(&Self::stat(b)))
                .map(|(i, _)| Action::Sell(i));
        }
        if self.p.board.len() >= 5
            && self.spare_hand() >= 2
            && let Some((i, _)) = self
                .p
                .board
                .iter()
                .enumerate()
                .find(|(i, m)| is(m, SHARK) && supply(m) && legal(Action::Sell(*i)))
        {
            if let Some(a) = self.position(false, legal) {
                return Some(a);
            }
            return Some(Action::Sell(i));
        }
        if !full
            || self
                .p
                .board
                .iter()
                .any(|m| !self.protects(m) && Self::stat(m) < 13.0)
        {
            return self
                .p
                .shop
                .iter()
                .enumerate()
                .filter(|(i, s)| s.item.as_minion().is_some_and(supply) && legal(Action::Buy(*i)))
                .filter(|(i, _)| {
                    game.slot_price(seat, *i)
                        .is_some_and(|(gold, health, _)| !health && gold <= 3)
                })
                .max_by_key(|(_, s)| {
                    self.card_value(s.item.card()) as i32
                        + s.item.as_minion().map_or(0, |m| self.active_supply(m))
                })
                .map(|(i, _)| Action::Buy(i));
        }
        None
    }
}

impl BuildPlan for Plan<'_> {
    fn protects(&self, m: &Minion) -> bool {
        self.route.spec().essential(id(m))
            && self.best_copy(m)
            && !(self.pressure()
                && !self.ready
                && id(m) != self.route.spec().anchor
                && Self::stat(m) < 4.5)
            || self.route == Tidecaller
                && is(m, APPRENTICE)
                && self.has(EYE)
                && self.p.board.iter().filter(|m| is(m, APPRENTICE)).count() <= 2
    }
    fn keep_bonus(&self, m: &Minion) -> f64 {
        let value = self.card_value(id(m))
            * if self.best_copy(m) || self.route == Shield && is(m, CZAR) {
                1.0
            } else {
                0.26
            };
        value
            + match self.route {
                Poison if tech(m) => 19.0 + f64::from(m.has(Kw::DIVINE_SHIELD)) * 5.0,
                Hope | Shield if m.has(Kw::DIVINE_SHIELD) => 8.0,
                DeityDeath | Sludge if m.is(Races::ABERRATION) => {
                    5.0 + f64::from(m.has(Kw::REBORN)) * 3.0
                }
                Lobster | BeastSummon | Behemoth | Beetle
                    if m.is(Races::BEAST) && m.has(Kw::REBORN) =>
                {
                    5.0
                }
                Magicfin if is(m, APPRENTICE) && self.has(EYE) => 22.0,
                Tidecaller if m.is(Races::MURLOC) => 3.0,
                _ => 0.0,
            }
    }
    fn acquisition_bonus(&self, m: &Minion, copies: usize) -> f64 {
        let tribe = self.route.race();
        let useful = if self.priority_piece(m) { 9.0 } else { 0.0 };
        let synergy = match self.route {
            Tempest | Ultraviolet if m.is(Races::ELEMENTAL) && self.ready => 6.0,
            DragonCry if self.ready && script(m.card).is_some_and(|s| s.battlecry.is_some()) => 5.0,
            Volumizer | MechSpell if m.has(Kw::MAGNETIC) => 9.0,
            Shield if m.has(Kw::DIVINE_SHIELD) => 9.0,
            DemonShop if m.is(Races::DEMON) && self.has("BG21_004") => 4.0,
            _ if m.is(tribe) && self.ready => 2.0,
            _ => 0.0,
        };
        let missing_tier_six = self.p.tier == 5
            && self
                .route
                .spec()
                .full
                .iter()
                .chain(std::iter::once(&self.route.spec().anchor))
                .any(|&c| db().get(c).tier == 6 && !has(&self.p.board, c));
        useful
            + synergy
            + f64::from(self.active_supply(m)) * 0.6
            + if missing_tier_six && copies == 2 && !self.pressure() {
                9.0
            } else {
                0.0
            }
            - if is(m, "BG26_525") && !self.has_rewind() && self.pressure() {
                35.0
            } else {
                0.0
            }
    }
    fn spell_bonus(&self, card: CardId) -> f64 {
        let key = key(card);
        let spell = script(card).map(|s| s.spell_target);
        let targeted = spell.is_some_and(|s| {
            matches!(
                s,
                TargetSpec::Friendly
                    | TargetSpec::FriendlyRace(_)
                    | TargetSpec::FriendlyOrTavern
                    | TargetSpec::AnyNoDivineShield
            )
        });
        match self.route {
            Magicfin
                if self.has(FUNGUS)
                    && matches!(
                        key,
                        "BG28_805" | "BG28_827" | "BG28_810" | "BG28_800" | "BG28_GIL_836"
                    ) =>
            {
                20.0
            }
            UndeadAttack | Knight if key == "BG28_604" => 11.0,
            HandGems | Barrage if key == "BG20_GEM" || key == "BG34_689" || key == "BG28_698" => {
                13.0
            }
            Shield | DeityDeath if is_tavern_spell(card) => 5.0,
            DemonShop | DemonDeath
                if matches!(key, "BG28_607" | "BG36_880" | "BG34_689" | "BG34_444") =>
            {
                13.0
            }
            DemonShop | Tidecaller | MechSpell | HandGems if targeted && self.ready => 7.0,
            _ => 0.0,
        }
    }
    fn hand_priority(&self, item: &Item) -> f64 {
        match item {
            Item::Minion(m) if self.priority_piece(m) => 55.0,
            Item::Minion(m) if is(m, APPRENTICE) && self.route == Magicfin => 30.0,
            Item::Spell { card, .. } if self.spell_bonus(*card) >= 8.0 => 15.0,
            _ => 0.0,
        }
    }
    fn can_cast(&self, card: CardId) -> bool {
        let name = key(card);
        !(self.route == Sludge
            && self.spare_hand() >= 2
            && matches!(name, "BG36_301t" | "BG36_371" | "BG36_303")
            && (self.has("BG36_320") || self.has("BG36_300") || self.has("BG36_312")))
            && !(self.route == Magicfin && name == "BG28_805" && self.spare_hand() < 1)
    }
    fn activate(&self, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        self.p
            .board
            .iter()
            .enumerate()
            .filter(|(i, m)| legal(Action::Activate(*i)) && self.usable(m))
            .find(|(_, m)| {
                matches!(
                    key(m.card),
                    KELP | "BG36_243"
                        | SALVAGER
                        | "BG36_362"
                        | "BG36_300"
                        | "BG36_312"
                        | "BG28_582"
                        | GEOMANT
                )
            })
            .map(|(i, _)| Action::Activate(i))
    }

    fn pending(&self, menu: &Pending, legal: &dyn Fn(Action) -> bool) -> Option<Action> {
        let Pending::Target { options, cont } = menu else {
            return None;
        };
        let source = match cont {
            Continuation::Activate { eid } | Continuation::Battlecry { eid, .. } => {
                key(self.p.board.iter().find(|m| m.eid == *eid)?.card)
            }
            Continuation::HandSpell { hid, .. } => {
                key(self.p.hand.iter().find(|h| h.hid == *hid)?.item.card())
            }
            _ => return None,
        };
        if matches!(source, "BG36_300" | "BG36_312" | "BG28_582") {
            let discard = |item: &Item| match item {
                Item::Spell { card, .. }
                    if matches!(key(*card), "BG36_301t" | "BG36_371" | "BG36_303") =>
                {
                    90.0
                }
                Item::Minion(m) => -Self::stat(m) - self.card_value(id(m)),
                Item::Spell { card, .. } => -8.0 - self.spell_bonus(*card),
            };
            return options
                .iter()
                .enumerate()
                .filter(|(i, _)| legal(Action::Choose(*i)))
                .filter_map(|(i, &eid)| {
                    self.p
                        .hand
                        .iter()
                        .find(|h| h.hid == eid)
                        .map(|h| (i, discard(&h.item)))
                })
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| Action::Choose(i));
        }
        if source == LIONFISH {
            return options
                .iter()
                .enumerate()
                .filter(|(i, _)| legal(Action::Choose(*i)))
                .filter_map(|(i, &eid)| {
                    self.p.shop.iter().find(|s| s.sid == eid).map(|s| {
                        (
                            i,
                            s.item
                                .as_minion()
                                .map_or(7.0, |m| Self::stat(m) + self.card_value(id(m))),
                        )
                    })
                })
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| Action::Choose(i));
        }
        let minion_score = |m: &Minion| -> Option<f64> {
            let score = match source {
                KELP => return Some(self.battlecry_target(m)),
                "BG36_243" => return Some(self.rally_target(m)),
                "BG36_511" | "BG28_604" => return Some(self.toss_value(m)),
                SALVAGER => {
                    if m.has(Kw::VENOMOUS) || m.has(Kw::DIVINE_SHIELD) {
                        16.0
                    } else {
                        6.0
                    }
                }
                "BG28_698" => 2.0 * f64::from(m.gems).sqrt() + self.spell_host(m),
                "BG28_607" | "BG36_880" | "BG23_357" | "BG20_GEM" | "BG36_624" | "BG28_507"
                | "BG28_897" | "BG32_337" | "BG36_883" | "BG28_520" => self.spell_host(m),
                "EBG_Spell_017" if self.priority_piece(m) || self.route.spec().essential(id(m)) => {
                    20.0
                }
                _ => return None,
            };
            Some(score + Self::stat(m).min(45.0) * 0.55)
        };
        options
            .iter()
            .enumerate()
            .filter(|(i, _)| legal(Action::Choose(*i)))
            .filter_map(|(i, &eid)| {
                self.p
                    .board
                    .iter()
                    .find(|m| m.eid == eid)
                    .and_then(|m| minion_score(m).map(|value| (i, value)))
            })
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| Action::Choose(i))
    }
    fn upgrade(&self, _: &Game, _: usize, ordinary: bool) -> bool {
        ordinary
    }
    fn search_missing(&self, _: &Game, _: usize) -> bool {
        false
    }
    fn attack_bonus(&self, m: &Minion) -> f64 {
        match self.route {
            Poison => {
                if m.has(Kw::VENOMOUS) || m.has(Kw::POISONOUS) {
                    7.0
                } else if is(m, "BG23_318") {
                    1.5
                } else {
                    4.0
                }
            }
            UndeadAttack | Knight if is(m, "BG36_515") => -100.0,
            UndeadAttack | Knight
                if is(m, "BG25_010") || is(m, "BG28_309") || is(m, "BG25_008") =>
            {
                23.0
            }
            Behemoth | Beetle | BeastSummon
                if is(m, RAT) || is(m, "BG31_809") || is(m, "BG31_801") =>
            {
                14.0
            }
            Behemoth | Beetle | BeastSummon if is(m, APE) || is(m, "BG35_602") => -27.0,
            Beetle if is(m, "BG36_209") => -30.0,
            DeityDeath if is(m, "BG36_318") => 15.0,
            DemonDeath if is(m, "BG36_362") => 11.0,
            _ if m.has(Kw::CLEAVE) => 12.0,
            _ => 0.0,
        }
    }
}
