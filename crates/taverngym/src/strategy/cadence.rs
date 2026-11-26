use crate::cards::{CardId, db};
use crate::script::script;
use crate::state::{Item, Minion, Pending, Player};
use crate::types::Kw;
use crate::{Action, Game, cardid};

use super::{apm, azerite, catalog_plan, demon, guide_bot::Build, quilboar, tea};

pub struct Cadence<'a> {
    player: &'a Player,
    turn: u32,
    danger: bool,
    tempo: f64,
    rush_six: bool,
}

impl<'a> Cadence<'a> {
    pub fn new(game: &'a Game, seat: usize) -> Self {
        let player = &game.players[seat];
        let health = player.health + player.armor;
        let tempo = strength(&player.board);
        let recent_threat = player.next_opponent.is_some_and(|opponent| {
            let (board, seen_turn) = game.known_board(seat, opponent);
            seen_turn > 0
                && game.turn.saturating_sub(seen_turn) <= 3
                && strength(board) > tempo * 1.45 + 10.0
        });
        let danger = game.turn >= 6
            && (health <= 13 || health <= 23 && (player.stats.last_damage <= -8 || recent_threat));
        Self {
            player,
            turn: game.turn,
            danger,
            tempo,
            rush_six: false,
        }
    }

    pub fn with_six_rush(mut self, enabled: bool) -> Self {
        self.rush_six = enabled && self.turn >= 6 && (3..=5).contains(&self.player.tier);
        self
    }

    pub fn rush_upgrade(&self, game: &Game, seat: usize) -> bool {
        self.rush_six && game.can_upgrade(seat)
    }

    pub fn reserve_six_refreshes(&self) -> bool {
        self.rush_six && self.player.tier == 5 && self.player.free_refreshes > 0 && !self.danger
    }

    pub fn rush_cash_sale(&self, game: &Game, seat: usize) -> Option<Action> {
        let p = self.player;
        if !self.rush_six || !p.stats.lost_last || p.gold >= game.upgrade_price(seat) {
            return None;
        }
        p.board
            .iter()
            .position(|m| {
                m.is_card_id(cardid!("BG24_018"))
                    && minion_strength(m) <= 14.0
                    && p.gold + if m.golden { 10 } else { 5 } >= game.upgrade_price(seat)
            })
            .map(Action::Sell)
    }

    pub fn wants_early_gift(&self, game: &Game, seat: usize) -> bool {
        if !game.can_dark_gift(seat) || !(3..=5).contains(&self.turn) {
            return false;
        }
        let pl = self.player;
        // At T5 the standard seven gold can pay for both. When the budget has
        // changed, prefer the level on an already adequate board.
        !(self.turn == 5
            && pl.tier == 2
            && pl.board.len() >= 4
            && pl.gold >= game.upgrade_price(seat)
            && pl.gold < game.upgrade_price(seat) + game.cfg.dark_gift_cost)
    }

    pub fn wants_upgrade(
        &self,
        game: &Game,
        seat: usize,
        build: Build,
        ordinary: bool,
        search_for_pivot: bool,
    ) -> bool {
        if self.rush_six {
            return game.can_upgrade(seat);
        }
        let pl = self.player;
        let price = game.upgrade_price(seat);
        let spare = pl.gold >= price + game.minion_price(seat);
        let health = pl.health + pl.armor;
        match pl.tier {
            1 => self.turn >= 2 && (!pl.board.is_empty() || self.turn >= 3 || spare),
            2 => self.turn >= 5 && (pl.board.len() >= 2 || spare),
            3 => {
                self.turn >= 6
                    && (spare || !self.danger && pl.board.len() >= 3 && self.tempo >= 24.0)
            }
            4 => {
                self.turn >= 8
                    && (spare || !self.danger && pl.board.len() >= 4 && self.tempo >= 36.0)
            }
            5 => {
                let azerite_board = azerite::pieces(&pl.board);
                let tea_board = tea::pieces(&pl.board);
                let needs_six = match build {
                    Build::Tea => !tea::has_core(tea_board),
                    Build::Azerite => {
                        azerite_board & azerite::AZERITE != 0
                            && (azerite_board.count_ones() >= 3
                                || pl.stats.tavern_elemental_hp >= 24)
                            && azerite_board != azerite::FULL
                    }
                    Build::General => true,
                    Build::Quilboar => quilboar::pieces(&pl.board) & quilboar::RIDER == 0,
                    Build::Demon => demon::needs_six(&pl.board),
                    Build::Apm(_) => true,
                    Build::Catalog(route) => {
                        catalog_plan::Plan::new(pl, route, self.turn).needs_six(game.lobby_races)
                    }
                };
                // Preserve a fifth-tier search window unless the board and wallet
                // can afford an early six, or a later discounted level is safe.
                let bridge_can_level = search_for_pivot
                    && health >= 24
                    && self.tempo >= 100.0
                    && pl.stats.last_damage >= 0
                    && pl.gold - price >= game.minion_price(seat) + game.refresh_price(seat);
                (needs_six || bridge_can_level)
                    && self.turn >= 10
                    && !self.danger
                    && pl.board.len() >= 5
                    && (health >= 23 && self.tempo >= 80.0 && (spare || pl.stats.last_damage >= 0)
                        || self.turn >= 11 && spare && self.tempo >= 65.0
                        || self.turn >= 12 && price <= 4 && self.tempo >= 85.0)
            }
            _ => ordinary,
        }
    }

    pub fn shop_bonus(&self, item: &Item) -> f64 {
        if self.turn == 1 && self.player.board.is_empty() && self.player.hand.is_empty() {
            return if item.as_minion().is_some() {
                20.0
            } else {
                -20.0
            };
        }
        match item {
            Item::Spell { card, .. } => self.economy_spell(*card),
            Item::Minion(m)
                if (6..=8).contains(&self.turn)
                    && self.player.tier < 5
                    && !self.danger
                    && self.player.board.len() >= 4
                    && m.tier <= 3 =>
            {
                // A healthy board can let a weak early-game sidegrade pass.
                let body = strength(std::slice::from_ref(m));
                if body < 10.0 && gift_value(m) == 0.0 {
                    -4.0
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    fn economy_spell(&self, card: CardId) -> f64 {
        match db().get(card).id.as_str() {
            "BG28_827" => {
                if self.player.tier < 5 {
                    7.0
                } else {
                    9.0
                }
            } // Two banked refreshes.
            "BG28_800" if !self.danger => 4.0, // Two gold next turn.
            "BG28_805" if self.turn <= 8 && !self.danger => 3.5, // Maximum gold.
            "BG28_571" if self.player.health + self.player.armor >= 19 && !self.danger => 7.0,
            _ => 0.0,
        }
    }

    pub fn pre_level_buy(
        &self,
        game: &Game,
        seat: usize,
        index: usize,
        score: f64,
        build: Build,
    ) -> bool {
        let pl = self.player;
        let item = &pl.shop[index].item;
        if build == Build::Demon
            && item.as_minion().is_some_and(|m| {
                demon::Plan::new(pl, self.turn).priority_piece(m)
                    && (self.turn != 2 || !pl.board.is_empty())
            })
        {
            return score >= 12.0;
        }
        if let Build::Apm(route) = build
            && item
                .as_minion()
                .is_some_and(|m| apm::Plan::new(pl, route).priority_piece(m))
        {
            return score >= 15.0;
        }
        if let Build::Catalog(route) = build
            && self.turn >= 6
            && item
                .as_minion()
                .is_some_and(|m| catalog_plan::Plan::new(pl, route, self.turn).priority_piece(m))
        {
            return score
                >= if self.turn <= 7 && pl.gold - game.upgrade_price(seat) < 3 {
                    31.0
                } else {
                    18.0
                };
        }
        if !(6..=10).contains(&self.turn) {
            return false;
        }
        let Some((cost, health_cost, _)) = game.slot_price(seat, index) else {
            return false;
        };
        let price = game.upgrade_price(seat);
        let spare = pl.gold - cost >= price;
        if let Item::Spell { card, .. } = item {
            return self.economy_spell(*card) >= 7.0
                && (spare || cost == 0 || health_cost && !self.danger)
                || spare && score >= 13.0;
        }
        let Some(minion) = item.as_minion() else {
            return false;
        };
        let key = ["BG28_707", "BG36_640", "BG35_883"].iter().any(|id| {
            minion.is_card_id(db().id(id)) && !pl.board.iter().any(|m| m.is_card_id(db().id(id)))
        });
        let rider = build == Build::Quilboar
            && minion.is_card_id(cardid!("BG31_323"))
            && quilboar::pieces(&pl.board) & quilboar::RIDER == 0;
        score >= 17.0
            && (spare
                || key
                || rider
                || self.danger && strength(std::slice::from_ref(minion)) >= 15.0)
    }

    pub fn before_rush_upgrade(
        &self,
        game: &Game,
        seat: usize,
        index: usize,
        score: f64,
        build: Build,
        replacement: Option<usize>,
    ) -> bool {
        let p = self.player;
        let Some((price, health_cost, _)) = game.slot_price(seat, index) else {
            return false;
        };
        let spare = health_cost || p.gold - price >= game.upgrade_price(seat);
        match &p.shop[index].item {
            Item::Spell { card, .. } => {
                let health = p.health + p.armor;
                match db().get(*card).id.as_str() {
                    "BG28_810" if !health_cost && price <= 1 => true, // Immediate reimbursement.
                    "BG28_571" if health_cost && health - price >= 17 => true,
                    "BG28_800" if price <= 1 && (!self.danger || spare) => true,
                    "BG28_805"
                        if self.turn <= 8
                            && !self.danger
                            && (spare
                                || price <= 2
                                || self.turn == 6 && price <= 3 && health >= 26) =>
                    {
                        true
                    }
                    "BG28_827"
                        if p.free_refreshes < 4
                            && (spare || p.tier < 5 && p.free_refreshes < 2) =>
                    {
                        true
                    }
                    _ => spare && score >= 17.0,
                }
            }
            Item::Minion(m) => {
                let key = db().get(m.normal_card()).id.as_str();
                let same = |b: &Minion| !b.golden && b.normal_card() == m.normal_card();
                let triple = !m.golden
                    && p.board.iter().filter(|b| same(b)).count()
                        + p.hand
                            .iter()
                            .filter(|h| h.locked == 0 && h.item.as_minion().is_some_and(same))
                            .count()
                        == 2;
                let missing = !apm::has(&p.board, key)
                    && !p.hand.iter().any(|h| {
                        h.item
                            .as_minion()
                            .is_some_and(|b| b.normal_card() == m.normal_card())
                    });
                let core = missing
                    && (matches!(key, "BG28_707" | "BG36_640" | "BG35_883" | "BG36_523")
                        || match build {
                            Build::Apm(route) => apm::Plan::new(p, route).priority_piece(m),
                            Build::Catalog(route) => {
                                catalog_plan::Plan::new(p, route, self.turn).priority_piece(m)
                            }
                            Build::Demon => demon::Plan::new(p, self.turn).priority_piece(m),
                            Build::Quilboar => m.is_card_id(cardid!("BG31_323")),
                            _ => false,
                        });
                let economy = !health_cost
                    && (key == "BG24_018" && p.stats.lost_last && price < 5
                        || key == "BG26_135" && (price <= 1 || apm::has(&p.board, "BG_LOE_077"))
                        || p.tier < 5
                            && p.free_refreshes < 2
                            && price <= 3
                            && refresh_deathrattle(m));
                let tempo =
                    minion_strength(m) - replacement.map_or(0.0, |i| minion_strength(&p.board[i]));
                let strong = if self.danger {
                    12.0_f64.max(self.tempo * 0.16)
                } else {
                    19.0_f64.max(self.tempo * 0.26)
                };
                triple && score >= 14.0
                    || core && score >= 10.0
                    || economy && (replacement.is_none() || spare)
                    || tempo >= strong && score >= 8.0
                    || spare && score >= 17.0
            }
        }
    }

    pub fn can_refresh(&self, game: &Game, seat: usize) -> bool {
        let p = self.player;
        if p.tier >= 5 {
            // Spend stored refreshes in a shop that can contain key pieces.
            return p.gold >= game.minion_price(seat) + game.refresh_price(seat)
                || p.free_refreshes > 0 && p.gold >= 1;
        }
        p.free_refreshes == 0 || self.danger && p.gold >= game.minion_price(seat)
    }

    pub fn can_power(&self, game: &Game, seat: usize, slot: usize) -> bool {
        let Some(power) = self.player.powers.get(slot) else {
            return false;
        };
        if power.card == cardid!("TB_BaconShop_HP_074")
            && power.counter % 4 == 3
            && (7..=9).contains(&self.turn)
            && self.player.tier < 5
            && !self.danger
        {
            return false; // Hold only the final dig if we are near tier five.
        }
        if power.card == cardid!("TB_BaconShop_HP_049")
            && self.danger
            && self.player.health + self.player.armor <= 8
        {
            return false;
        }
        // Extra hero power slots and temporary price adjustments follow engine legality.
        game.hero_power_price(seat, slot).is_some()
    }

    pub fn trinket(
        &self,
        game: &Game,
        seat: usize,
        menu: &Pending,
        legal: &impl Fn(Action) -> bool,
    ) -> Option<Action> {
        let Pending::Trinket { options, free, .. } = menu else {
            return None;
        };
        if self.turn > 8 || *free {
            return None;
        }
        let upgrade = game.upgrade_price(seat);
        let score = |card: CardId| {
            let price = game.trinket_price(seat, card);
            let premium = match db().get(card).id.as_str() {
                "BG30_MagicItem_847" | "BG32_MagicItem_428" | "BG36_MagicItem_220" => 9,
                "BG30_MagicItem_998" | "BG35_MagicItem_818" | "BG36_MagicItem_411" => 9,
                "BG30_MagicItem_430" => 6,
                "BG30_MagicItem_902" if self.player.board.len() >= 3 => 7,
                _ => 0,
            };
            let room_to_level =
                !self.danger && self.player.tier < 5 && self.player.gold - price >= upgrade;
            premium - price + 3 * i32::from(room_to_level)
        };
        options
            .iter()
            .enumerate()
            .filter(|(i, _)| legal(Action::Choose(*i)))
            .max_by_key(|(i, card)| (score(**card), std::cmp::Reverse(*i)))
            .map(|(i, _)| Action::Choose(i))
    }
}

pub fn gift_value(m: &Minion) -> f64 {
    m.attached
        .iter()
        .map(|&card| match db().get(card).id.as_str() {
            "BG36_MidGameEffect_000t82" => 13.0, // Affinity: a card every two turns.
            "BG36_MidGameEffect_000t52" => 12.0 + if m.has(Kw::REBORN) { 3.0 } else { 0.0 },
            "BG36_MidGameEffect_000t18" => 10.0, // Replication.
            "BG36_MidGameEffect_000t51" | "BG36_MidGameEffect_000t11" => 10.0,
            "BG36_MidGameEffect_000t13" => 9.0,
            "BG36_MidGameEffect_000t73"
            | "BG36_MidGameEffect_000t74"
            | "BG36_MidGameEffect_000t75" => 7.0,
            "BG36_MidGameEffect_000t5" | "BG36_MidGameEffect_000t14" => 6.0,
            "BG36_MidGameEffect_000t10"
                if script(m.card).is_some_and(|s| s.battlecry.is_some()) =>
            {
                7.0
            }
            _ => 0.0,
        })
        .sum()
}

pub fn refresh_deathrattle(m: &Minion) -> bool {
    m.attached.contains(&cardid!("BG36_MidGameEffect_000t52"))
}

pub fn strength(board: &[Minion]) -> f64 {
    board.iter().map(minion_strength).sum()
}

pub fn minion_strength(m: &Minion) -> f64 {
    let attack = f64::from(m.atk.max(0));
    let mut value = attack + f64::from(m.hp.max(0));
    if m.has(Kw::DIVINE_SHIELD) {
        value += 3.0 + 0.5 * attack;
    }
    if m.has(Kw::REBORN) {
        value += 2.0;
    }
    if m.has(Kw::VENOMOUS) {
        value += 6.0;
    }
    if script(m.card).is_some_and(|s| s.deathrattle.is_some()) {
        value += 2.0;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Phase;
    use crate::state::ShopSlot;

    fn setup() -> Game {
        let mut game = Game::new(17, crate::RulesConfig::default());
        game.phase = Phase::Recruit;
        game.turn = 9;
        let p = &mut game.players[0];
        p.pending.clear();
        p.tier = 5;
        p.gold = 10;
        p.upgrade_cost = 10;
        p.health = 10;
        p.armor = 0;
        p.stats.last_damage = -8;
        game
    }

    #[test]
    fn rush_levels_with_exact_gold_even_on_a_weak_board() {
        let mut game = setup();
        let wants = |game: &Game, rush| {
            Cadence::new(game, 0).with_six_rush(rush).wants_upgrade(
                game,
                0,
                Build::General,
                false,
                false,
            )
        };
        assert!(wants(&game, true));
        assert!(!wants(&game, false));
        game.players[0].gold = 9;
        assert!(!wants(&game, true));
        game.players[0].gold = 10;
        game.turn = 5;
        assert!(!wants(&game, true));
    }

    #[test]
    fn rush_distinguishes_a_sidegrade_from_real_tempo_and_economy() {
        let mut game = setup();
        game.players[0].health = 30;
        let cheap = Minion::new(101, cardid!("BG25_001"));
        let mut big = Minion::new(102, cardid!("BG25_001"));
        big.atk = 12;
        big.hp = 12;
        game.players[0].shop = vec![
            ShopSlot::new(201, Item::Minion(cheap)),
            ShopSlot::new(202, Item::Minion(big.clone())),
            ShopSlot::new(203, Item::spell(cardid!("BG28_800"))),
            ShopSlot::new(204, Item::Minion(Minion::new(104, cardid!("BG36_523")))),
        ];
        game.players[0].board.push(big);
        let timing = Cadence::new(&game, 0).with_six_rush(true);
        assert!(!timing.before_rush_upgrade(&game, 0, 0, 14.0, Build::General, None));
        assert!(timing.before_rush_upgrade(&game, 0, 1, 14.0, Build::General, None));
        assert!(!timing.before_rush_upgrade(&game, 0, 1, 14.0, Build::General, Some(0)));
        assert!(timing.before_rush_upgrade(&game, 0, 2, 14.0, Build::General, None));
        assert!(timing.before_rush_upgrade(&game, 0, 3, 14.0, Build::General, None));
        game.players[0]
            .board
            .push(Minion::new(105, cardid!("BG25_001")));
        let timing = Cadence::new(&game, 0).with_six_rush(true);
        assert!(timing.before_rush_upgrade(&game, 0, 0, 14.0, Build::General, None));
    }
}
